"""Exercise the renderer's real sampler without opening an SDL window."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def telemetry_probe(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    source = (REPO / "src/render/rSysdep.cpp").read_text()
    start = source.index("namespace\n{\nbool sr_frameTelemetry = false;")
    end = source.index("\nvoid rSysDep::SwapGL()", start)
    # Compile the actual implementation against deterministic clock/SDL/file-system stubs.
    # No copy of the percentile, cadence, frame classification or output logic lives here.
    prelude = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#define SDL_VERSION_ATLEAST(a,b,c) 1
using Uint32 = uint32_t;
constexpr Uint32 SDL_WINDOW_FULLSCREEN=1, SDL_WINDOW_FULLSCREEN_DESKTOP=4097;
constexpr Uint32 SDL_WINDOW_INPUT_FOCUS=512, SDL_WINDOW_HIDDEN=8, SDL_WINDOW_MINIMIZED=64;
constexpr int SDL_GL_MULTISAMPLEBUFFERS=1, SDL_GL_MULTISAMPLESAMPLES=2;
struct SDL_DisplayMode { int refresh_rate=120; };
int *sr_screen=nullptr;
bool sr_hiddenWindow=false, s_videoout=false;
int sr_screenWidth=100, sr_screenHeight=100, sr_maxFPS=240, sr_floorMirror=2;
double clockNow=10;
int ioOpens=0, sdlQueries=0;
Uint32 windowFlags=SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_INPUT_FOCUS;
std::string logPath;
struct { int vSync=2, multisamples=4; bool fullscreen=false; } currentScreensetting;
template<class T> struct tConfItem { tConfItem(char const *, T &) {} };
struct rSysDep { static int swapOptimize_; static void GameFrameRendered(double, int); };
int rSysDep::swapOptimize_=1;
double tRealSysTimeFloat() { return clockNow; }
std::string st_GetCurrentTime(char const *) { return "2026-10-19T12:00:00+1100"; }
void SDL_GL_GetDrawableSize(int *,int *w,int *h) { ++sdlQueries; *w=3024; *h=1964; }
void SDL_GetWindowSize(int *,int *w,int *h) { ++sdlQueries; *w=1512; *h=982; }
Uint32 SDL_GetWindowFlags(int *) { ++sdlQueries; return windowFlags; }
int SDL_GetWindowDisplayIndex(int *) { ++sdlQueries; return 0; }
int SDL_GetCurrentDisplayMode(int,SDL_DisplayMode *) { ++sdlQueries; return 0; }
int SDL_GL_GetSwapInterval() { ++sdlQueries; return 0; }
int SDL_GL_GetAttribute(int key,int *out) { ++sdlQueries; *out=key==SDL_GL_MULTISAMPLEBUFFERS ? 1 : 2; return 0; }
namespace tDirectories {
struct Path { bool Open(std::ofstream &file,char const *name,std::ios::openmode mode) const {
    assert(std::string(name)=="frame-times.log" && (mode & std::ios::app));
    ++ioOpens; file.open(logPath,mode); return bool(file);
} };
Path const &Var() { static Path path; return path; }
}
'''
    main = r'''
int main(int argc,char **argv) {
    assert(argc==3);
    logPath=argv[2];
    std::string mode=argv[1];
    assert(!FrameTelemetryEnabled());
    rSysDep::GameFrameRendered(0,59);
    assert(!sr_hiddenGameFramePending && ioOpens==0 && sdlQueries==0);
    if(mode=="off") return 0;
    sr_hiddenWindow=mode=="hidden";
    sr_frameTelemetry=mode=="visible";
    assert(FrameTelemetryEnabled());
    HiddenFrameTelemetry telemetry;
    auto present=[&](double now,bool game,double time) {
        clockNow=now;
        if(game) rSysDep::GameFrameRendered(time,59);
        telemetry.Present(sr_hiddenGameFramePending,sr_hiddenGameTime,sr_hiddenReportedFPS);
        sr_hiddenGameFramePending=false;
    };
    present(10,true,0);
    present(10.010,true,.010);
    present(10.050,false,0);
    present(14.990,true,4.990);
    assert(ioOpens==0 && sdlQueries==0);
    present(15.010,true,5.010);
    assert(ioOpens==(sr_frameTelemetry ? 1 : 0) && sdlQueries==8);
    present(15.020,true,5.020);
    assert(ioOpens==(sr_frameTelemetry ? 1 : 0) && sdlQueries==8);
    present(20.020,true,10.020);
    assert(ioOpens==(sr_frameTelemetry ? 2 : 0) && sdlQueries==16);
    return 0;
}
'''
    directory = tmp_path_factory.mktemp("frame-telemetry")
    cpp = directory / "probe.cpp"
    cpp.write_text(prelude + source[start:end] + main)
    executable = directory / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)], check=True)
    return executable


def test_normal_play_collects_nothing_without_opt_in(telemetry_probe, tmp_path):
    log = tmp_path / "frame-times.log"
    result = subprocess.run([str(telemetry_probe), "off", str(log)], check=True, capture_output=True, text=True)
    assert not result.stdout and not result.stderr and not log.exists()


@pytest.mark.parametrize("mode", ["hidden", "visible"])
def test_telemetry_preserves_hidden_output_and_appends_visible_frame_reports(telemetry_probe, tmp_path, mode):
    log = tmp_path / "frame-times.log"
    result = subprocess.run([str(telemetry_probe), mode, str(log)], check=True, capture_output=True, text=True)
    assert not result.stdout
    if mode == "hidden":
        assert not log.exists()
        text = result.stderr
        assert text.count("Hidden renderer:") == 2 and "Hidden frame times (active game)" in text
    else:
        assert not result.stderr
        text = log.read_text()
        assert text.count("Renderer:") == 2  # each report appends instead of replacing the file
    first = text.splitlines()[0]
    for expected in ("presentations=5", "game_frames=4", "non_game_presentations=1", "hud_fps=59",
                     "drawable=3024x1964", "window=1512x982", "display_mode_hz=120", "swap_interval=0",
                     "fullscreen_requested=0", "fullscreen=1", "fullscreen_desktop=1", "focused=1",
                     "hidden=0", "minimized=0", "msaa_requested=4", "msaa_buffers=1", "msaa_samples=2",
                     "timestamp=2026-10-19T12:00:00+1100", "floor_mirror=2"):
        assert expected in first
    assert re.search(r"\bpid=\d+\b", first)
    active = next(line for line in text.splitlines() if "frame times (active game)" in line.lower())
    assert "samples=2 p50_ms=10 p95_ms=20 p99_ms=20 worst_ms=20" in active
    assert "one_percent_low_fps=50" in active and "over_8_333ms=2/2" in active
    assert "active_game_fps=66.6667" in first  # the long menu gap is excluded
