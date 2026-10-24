#include "doctest.h"
#include "tSysTime.h"
#include <cmath>
#include "tMath.h"
#include <tuple>
#include <chrono>
#include "ConfItemTempValue.h"
#include "MockConsole.h"

// Tests for tSysTime utilities
// Purpose: Document the status quo behavior and detect regressions

TEST_SUITE("tSysTime")
{
    TEST_CASE("tTimerIsAccurate function")
    {
        GIVEN("system timer")
        {
            THEN("tTimerIsAccurate returns a boolean")
            {
                // This function checks if high-resolution timers are available
                bool isAccurate = tTimerIsAccurate();
                std::ignore = isAccurate;
            }
        }
    }

    TEST_CASE("tSysTimeFloat and tRealSysTimeFloat functions")
    {
        GIVEN("system time functions")
        {
            THEN("time functions return non-negative values")
            {
                double sysTime = tSysTimeFloat();
                double realSysTime = tRealSysTimeFloat();

                // Time should be non-negative (assuming system time is set correctly)
                CHECK(sysTime >= 0.0);
                CHECK(realSysTime >= 0.0);
            }

            THEN("time functions return finite values")
            {
                double sysTime = tSysTimeFloat();
                double realSysTime = tRealSysTimeFloat();

                // Time should be finite
                CHECK(std::isfinite(sysTime));
                CHECK(std::isfinite(realSysTime));
            }

            THEN("time increases between calls")
            {
                double time1 = tRealSysTimeFloat();
                double time2 = tRealSysTimeFloat();

                // Time should generally increase (though this might occasionally fail
                // if the calls happen in the same time quantum)
                // We use a weak check here
                CHECK(time2 >= time1);
            }
        }
    }

    TEST_CASE("tAdvanceFrame function")
    {
        GIVEN("frame advancement")
        {
            THEN("tAdvanceFrame can be called without crashing")
            {
                // Test with default delay
                tAdvanceFrame();

                // Test with explicit delay
                tAdvanceFrame(0);
                tAdvanceFrame(1000); // 1000 microseconds = 1ms
            }
        }
    }

    TEST_CASE("tDelay functions")
    {
        GIVEN("delay functions")
        {
            THEN("tDelay can be called without crashing")
            {
                // Test with small delay
                tDelay(100); // 100 microseconds
            }

            THEN("tDelayForce can be called without crashing")
            {
                // Test with small delay
                tDelayForce(100); // 100 microseconds
            }
        }
    }
}

// TODO: More comprehensive timing tests could be added, but they would require
// more complex setup and might be flaky in automated test environments.
// The current tests verify basic functionality without making strong assumptions
// about timing behavior.
namespace
{
struct ClockFixture
{
    MockConsole quiet;
    ConfItemTempValue<float> factor{"TIME_FACTOR"};
    ClockFixture()
    {
        factor.SetValue(1);
        tSetLockstep(0);
        tAdvanceFrame();
    }
    ~ClockFixture() { tSetLockstep(0); }
};
} // namespace

TEST_CASE("Training clock advances exactly and returns to live time without going backwards")
{
    ClockFixture fixture;
    auto wallStart = std::chrono::steady_clock::now();
    double realStart = tRealSysTimeFloat();
    double frameStart = tSysTimeFloat();
    tSetLockstep(.125);
    for (int i = 0; i < 16; ++i)
        tAdvanceFrame();
    double simulated = tSysTimeFloat();
    CHECK(simulated == doctest::Approx(frameStart + 2).epsilon(0.000001));
    tSetLockstep(0);
    tAdvanceFrame();
    CHECK(tSysTimeFloat() >= simulated);
    double realElapsed = tRealSysTimeFloat() - realStart;
    double wallElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    CHECK(realElapsed >= 0);
    CHECK(realElapsed <= wallElapsed + .025);
}

TEST_CASE("Upstream mock clock advances both timelines while preserving the training step")
{
    ClockFixture fixture;
    tSetLockstep(.125);
    double frameStart = tSysTimeFloat();
    double realStart = tRealSysTimeFloat();
    tMockAdvanceFrame(.5);
    CHECK(tSysTimeFloat() == doctest::Approx(frameStart + .5).epsilon(0.000001));
    CHECK(tRealSysTimeFloat() >= realStart + .499);
    tAdvanceFrame();
    CHECK(tSysTimeFloat() == doctest::Approx(frameStart + .625).epsilon(0.000001));
}

TEST_CASE("Upstream mock clock advances normal game time and wall time")
{
    ClockFixture fixture;
    double frameStart = tSysTimeFloat();
    double realStart = tRealSysTimeFloat();
    tMockAdvanceFrame(.25);
    CHECK(tSysTimeFloat() >= frameStart + .249);
    CHECK(tRealSysTimeFloat() >= realStart + .249);
    CHECK_FALSE(tLockstepActive());
}
