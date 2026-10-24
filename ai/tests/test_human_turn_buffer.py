"""Compile the real human dispatch and turn-buffer paths against deterministic stubs."""

import shutil
import subprocess
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def human_input_probe(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    cycle = (REPO / "src/tron/gCycle.cpp").read_text()
    movement = (REPO / "src/tron/gCycleMovement.cpp").read_text()
    player = (REPO / "src/engine/ePlayer.cpp").read_text()
    game_object = (REPO / "src/engine/eGameObject.h").read_text()
    cycle_header = (REPO / "src/tron/gCycle.h").read_text()

    local_input = function(cycle, "bool gCycle::ActLocalInput(")
    ordinary_input = function(cycle, "bool gCycle::Act(")
    default_input = function(game_object, "virtual bool ActLocalInput(")
    declaration_start = cycle_header.index("bool ActLocalInput(")
    declaration_end = cycle_header.index(";", declaration_start) + 1
    local_declaration = cycle_header[declaration_start:declaration_end]
    player = function(player, "bool ePlayer::Act(")
    dispatch_start = player.index("        bool objectAct = false;")
    dispatch_end = player.index("        return ret;", dispatch_start) + len("        return ret;")
    dispatch = player[dispatch_start:dispatch_end]

    # Exercise the actual legacy enqueue and dequeue blocks too. Collision,
    # acceleration, rendering and networking are replaced by small clock/counter
    # stubs; this probe verifies input routing and buffering, not full physics.
    turn = function(movement, "bool gCycleMovement::DoTurn(")
    enqueue = turn[turn.index("        int maxPendingTurns=sg_cycleTurnMemory;"):
                   turn.rindex("    return false;")]
    # The selected tail contains the closing brace of DoTurn's else block.
    enqueue = enqueue[:enqueue.rfind("    }")]
    anchor = movement.index("// simulate exactly to the time of the next turn if it is in reach")
    guard = re.search(r"(?m)^[ \t]*if\s*\(\s*!pendingTurns\.empty\(\)", movement[anchor:])
    assert guard, "queued-turn guard missing from the production timestep"
    dequeue_start = anchor + guard.start()
    dequeue_end = movement.index("    // do the rest of the timestep", dequeue_start)
    dequeue = movement[dequeue_start:dequeue_end]

    prelude = r'''
#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "gCycleInput.h"
using REAL = double;
struct uAction { int id; explicit uAction(int value):id(value){} };
struct uActionPlayer: uAction {
    explicit uActionPlayer(int value):uAction(value){}
    bool operator==(uActionPlayer const &other) const { return id==other.id; }
};
struct uActionCamera: uAction { explicit uActionCamera(int value):uAction(value){} };
REAL gameTime=0;
REAL se_GameTime() { return gameTime; }
enum { nSTANDALONE, nSERVER, nCLIENT };
int networkState=nSTANDALONE;
int sn_GetNetState() { return networkState; }
struct Timer { REAL speed=1; REAL Time() const { return gameTime; } };
Timer timer;
Timer *se_mainGameTimer=&timer;
int sg_cycleInputBuffer=1, sg_cycleTurnMemory=3;
REAL turnDelay=.1;
struct Joystick { bool Act(uActionPlayer *,REAL) { return false; } };
uActionPlayer s_brakeToggle(4);
class eGameObject {
public:
    static uActionPlayer se_turnLeft, se_turnRight;
    int ordinaryCalls=0;
    REAL lastValue=0;
    virtual ~eGameObject()=default;
    virtual bool Act(uActionPlayer *,REAL value) {
        ++ordinaryCalls; lastValue=value; return true;
    }
'''
    middle = r'''
};
uActionPlayer eGameObject::se_turnLeft(1), eGameObject::se_turnRight(2);
class gCycle: public eGameObject {
public:
    static uActionPlayer s_brake;
    Joystick *joystick_=nullptr;
    bool alive=true;
    bool arcActive_=false;
    unsigned short braking=0;
    REAL lastTime=0, nextTurnTime=0;
    int turnCalls=0, destinations=0, accelerationChanges=0, syncRequests=0;
    std::deque<int> pendingTurns;
    std::vector<std::pair<REAL,int>> executed;
    bool Alive() const { return alive; }
    std::size_t PendingTurnCount() const { return pendingTurns.size(); }
    void RequestSync(bool) { ++syncRequests; }
    void AccelerationDiscontinuity() { ++accelerationChanges; }
    void AddDestination() { ++destinations; }
    REAL GetNextTurn(int) const { return nextTurnTime; }
    void TimestepCore(REAL now) { lastTime=now; }
'''
    after_declaration = r'''
    bool Act(uActionPlayer *,REAL) override;
    bool Turn(int dir) {
        ++turnCalls;
        if(!Alive()) return false;
        if(nextTurnTime<=lastTime) {
            executed.emplace_back(lastTime,dir);
            nextTurnTime=lastTime+turnDelay;
            AddDestination();
            return true;
        }
'''
    after_enqueue = r'''
        return false;
    }
    void Advance(REAL currentTime) {
'''
    after_dequeue = r'''
        TimestepCore(currentTime);
        gameTime=currentTime;
    }
};
uActionPlayer gCycle::s_brake(3);
'''
    player_prelude = r'''
struct Camera {
    bool consumes=false;
    int calls=0;
    bool Act(uActionCamera *,REAL) { ++calls; return consumes; }
};
struct NetPlayer { int activity=0; void Activity() { ++activity; } };
bool se_chatter=false;
class ePlayer {
public:
    Camera *cam=nullptr;
    NetPlayer *netPlayer=nullptr;
    bool Dispatch(eGameObject *object,uAction *act,REAL x) {
'''
    after_dispatch = r'''
    }
};
'''
    source = (prelude + default_input + middle + local_declaration + after_declaration
              + enqueue + after_enqueue + dequeue
              + after_dequeue + local_input + "\n" + ordinary_input + "\n"
              + player_prelude + dispatch + after_dispatch)
    source += (Path(__file__).parent / "cpp/human_turn_buffer_check.cpp").read_text()
    directory = tmp_path_factory.mktemp("human-turn-buffer")
    cpp = directory / "probe.cpp"
    cpp.write_text(source)
    executable = directory / "probe"
    compile_run = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                                  "-I", str(REPO / "src/tron"), str(cpp), "-o", str(executable)],
                                 capture_output=True, text=True)
    assert compile_run.returncode == 0, compile_run.stderr
    return executable


@pytest.mark.parametrize("scenario", [
    "default-dispatch", "burst", "opposite", "cooldown-release", "braking",
    "bot-isolation", "classic", "existing-auto-drive", "camera-start-gates",
    "per-cycle-lifetime", "configured-cooldown",
    "active-arc-queue",
])
def test_real_human_dispatch_and_buffer_policy(human_input_probe, scenario):
    subprocess.run([str(human_input_probe), scenario], check=True, timeout=5)
