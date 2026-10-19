"""Native live-event clock fixtures, without SDL windows or a game process."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def input_timing_probe(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    executable = tmp_path_factory.mktemp("input-timing") / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(REPO / "ai/tests/cpp/input_timing_check.cpp"), "-o", str(executable)], check=True)
    return executable


@pytest.mark.parametrize("case", ["press_release", "turn_timeline", "speed_pause", "countdown_reset",
                                  "wrap", "clamp_order", "invalid_frame", "nonfinite"])
def test_native_live_event_time(input_timing_probe, case):
    result = subprocess.run([str(input_timing_probe), case], check=True, capture_output=True, text=True)
    assert result.stdout == "ok\n" and not result.stderr


def test_live_queue_records_mapped_times_and_playback_keeps_archived_times(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    source = (REPO / "src/ui/uInputQueue.cpp").read_text()
    header = (REPO / "src/ui/uInputQueue.h").read_text()
    callback = header[header.index("class su_TimerCallback{"):header.rindex("#endif")]
    timer = source[source.index("static su_TimerCallback *timer=NULL;"):source.index("bool su_prefetchInput=")]
    fetch = source[source.index("bool su_GetSDLInput(SDL_Event &tEvent,REAL &time){"):
                   source.index("\n/*\nint su_InputThread")]
    # Compile the actual queue retrieval/recording function with recorder and SDL stubs.
    # This checks branch ordering, not another copy of its timestamping implementation.
    prelude = r'''
#include "uInputTime.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>
using REAL = double;
struct SDL_Event { int type=0; struct { unsigned timestamp=0; } common; };
#define SDL_VERSION_ATLEAST(a,b,c) 1
struct tPlaybackBlock {};
struct tRecordingBlock {};
bool playback=false;
int pollCalls=0, lockCalls=0, eventCalls=0;
double recorded=0;
struct tRecorder { static bool Playback(char const *) { return false; }
                   static void Record(char const *) {} };
struct uMenu { static int quickexit; enum { QuickExit_Total=1 }; };
int uMenu::quickexit=0;
static char const *su_end="END";
bool su_markerRequired=false;
SDL_Event tEvents[100];
REAL times[100];
int current_out=0,currentIn=0;
bool input_get=false;
void increase(int &index) { index=(index+1)%100; }
void sr_LockSDL() { ++lockCalls; }
void sr_UnlockSDL() {}
int SDL_PollEvent(SDL_Event *event) {
    ++pollCalls;
    event->type=1;
    event->common.timestamp=pollCalls==1 ? 1025 : 1075;
    return 1;
}
template<class Mode> struct EventArchiver {
    static bool Archive(SDL_Event &event,REAL &time,bool &ret) {
        if constexpr(std::is_same<Mode,tPlaybackBlock>::value) {
            if(!playback) return false;
            event.type=1;
            time=-123.25;
            ret=true;
        } else recorded=time;
        return true;
    }
};
'''
    main = r'''
struct Clock: su_TimerCallback {
    uInputTimeMapper &mapper;
    explicit Clock(uInputTimeMapper &m): mapper(m) {}
    REAL GetTime() override { return mapper.CurrentTime(); }
    REAL GetEventTime(SDL_Event const &event) override {
        ++eventCalls;
        return mapper.EventTime(event.common.timestamp);
    }
};
void near(double a,double b) { assert(std::abs(a-b)<1e-9); }
int main() {
    uInputTimeMapper mapper;
    mapper.BeginFrame(1000,10,10);
    mapper.BeginFrame(1100,10.1,10);
    SDL_Event event;
    REAL time=0;
    {
        Clock clock(mapper);
        assert(su_GetSDLInput(event,time));
        near(time,10.025); near(recorded,time);
        assert(su_GetSDLInput(event,time));
        near(time,10.075); near(recorded,time);
        // Prefetched events must use their SDL timestamp, not the old cached game time.
        currentIn=1;
        tEvents[0].type=1;
        tEvents[0].common.timestamp=1080;
        times[0]=0;
        assert(su_GetSDLInput(event,time));
        near(time,10.08); near(recorded,time);
        assert(pollCalls==2 && eventCalls==3);
        // Replay bypasses live timing completely, including its clamps and event clock.
        playback=true;
        int locksBefore=lockCalls;
        assert(su_GetSDLInput(event,time));
        near(time,-123.25); near(recorded,time);
        assert(pollCalls==2 && eventCalls==3 && lockCalls==locksBefore);
        playback=false;
        {
            uInputTimeMapper nestedMapper;
            nestedMapper.BeginFrame(1100,20,20);
            Clock nested(nestedMapper);
            near(Time(),20);
        }
        near(Time(),10.1); // scoped clocks restore the outer callback
    }
    near(Time(),0); // menus without a game clock retain their existing behavior
}
'''
    cpp = tmp_path / "queue_probe.cpp"
    cpp.write_text(prelude + callback + timer + fetch + main)
    executable = tmp_path / "queue_probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(REPO / "src/ui"),
                    str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
