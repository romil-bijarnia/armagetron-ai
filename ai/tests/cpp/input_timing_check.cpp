#include "../../../src/ui/uInputTime.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

static void near(double actual, double expected)
{
    assert(std::isfinite(actual) && std::abs(actual - expected) < 1e-9);
}

static uInputTimeMapper frame(double start = 10, double finish = 10.1, double minimum = 10)
{
    uInputTimeMapper clock;
    clock.BeginFrame(1000, start, start);
    clock.BeginFrame(1100, finish, minimum);
    return clock;
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    std::string mode = argv[1];
    if (mode == "press_release")
    {
        auto clock = frame();
        double press = clock.EventTime(1025), release = clock.EventTime(1075);
        near(press, 10.025);
        near(release, 10.075);
        // An input held for 50 ms must retain that duration within a slower frame.
        near(release - press, .050);
    }
    else if (mode == "turn_timeline")
    {
        auto clock = frame();
        double first = clock.EventTime(1020), second = clock.EventTime(1080);
        // Independent kinematic fixture: drive at 100 units/s, turn left twice,
        // then finish this frame. Expected segments are 2, 6 and 2 units.
        double x = (first - 10) * 100 - (10.1 - second) * 100;
        double y = (second - first) * 100;
        near(x, 0);
        near(y, 6);
        // A legal 50 ms turn interval must not collapse to one timestamp.
        assert(second - first > .05);
        near(first, 10.02);
        near(second, 10.08);
    }
    else if (mode == "speed_pause")
    {
        for (double speed : {.25, .5, 1., 2., 4.})
        {
            auto clock = frame(10, 10 + .1 * speed);
            near(clock.EventTime(1025), 10 + .025 * speed);
            near(clock.EventTime(1075), 10 + .075 * speed);
        }
        auto clock = frame(10, 10);
        near(clock.EventTime(1020), 10);
        near(clock.EventTime(1080), 10);
        clock.BeginFrame(1200, 10.1, 10);
        near(clock.EventTime(1150), 10.05);
    }
    else if (mode == "countdown_reset")
    {
        auto clock = frame(-.02, .08, 0);
        near(clock.EventTime(1010), 0);
        near(clock.EventTime(1050), .03);
        clock.BeginFrame(1200, -4, 0); // timer reset without reusing the old epoch
        near(clock.EventTime(1150), -4);
        clock.Reset();
        clock.BeginFrame(1300, 2, 2);
        near(clock.EventTime(1250), 2);
        clock.BeginFrame(1400, 2.1, 2);
        near(clock.EventTime(1350), 2.05);
    }
    else if (mode == "wrap")
    {
        uInputTimeMapper clock;
        clock.BeginFrame(0xffffffe0u, 10, 10);
        clock.BeginFrame(0x44u, 10.1, 10);
        near(clock.EventTime(0xfffffff4u), 10.02);
        near(clock.EventTime(0x30u), 10.08);
    }
    else if (mode == "clamp_order")
    {
        auto clock = frame(10, 10.1, 10.03);
        near(clock.EventTime(900), 10.03); // stale event cannot rewind simulation
        near(clock.EventTime(1050), 10.05);
        near(clock.EventTime(1040), 10.05); // preserve queue order
        near(clock.EventTime(1080), 10.08);
        near(clock.EventTime(1120), 10.1); // future timestamp uses current frame
        near(clock.EventTime(1090), 10.1);
        clock = frame();
        near(clock.EventTime(0), 10.1); // missing timestamp
    }
    else if (mode == "invalid_frame")
    {
        auto clock = frame();
        clock.BeginFrame(1100, 10.2, 10.1); // no SDL tick elapsed
        near(clock.EventTime(1090), 10.2);
        clock.BeginFrame(1000, 10.3, 10.2); // ambiguous/reversed SDL clock
        near(clock.EventTime(990), 10.3);
        clock.BeginFrame(1200, 10.4, 10.3, false); // playback/lockstep/non-game
        near(clock.EventTime(1150), 10.4);
        clock.BeginFrame(1300, 10.5, 10.4); // first live frame has no old anchor
        near(clock.EventTime(1250), 10.5);
        clock.BeginFrame(1400, 10.6, 10.5);
        near(clock.EventTime(1350), 10.55);
    }
    else if (mode == "nonfinite")
    {
        auto clock = frame();
        clock.BeginFrame(1200, std::numeric_limits<double>::infinity(), 10);
        near(clock.EventTime(1150), 0);
        clock.BeginFrame(1300, 10, std::numeric_limits<double>::quiet_NaN());
        near(clock.EventTime(1250), 10);
    }
    else
        return 2;
    std::cout << "ok\n";
}
