#include "../../../src/tron/gLightcycleMotion.h"
#include <cassert>
#include <cmath>
#include <limits>

void near(double a, double b) { assert(std::abs(a - b) < 1e-8); }

int main()
{
    using namespace gLightcycleMotion;
    State motion;
    motion.Update(.99, 0, 0);
    motion.Update(1.005, Pi / 2, 1);
    assert(motion.Yaw() > 0 && motion.Yaw() < Pi / 18); // no 30-degree first-frame snap
    assert(motion.Bank() < 0);
    double firstYaw = motion.Yaw(), firstBank = motion.Bank();
    motion.Update(1.005, Pi / 2, 1); // reflection and main pass must agree
    near(motion.Yaw(), firstYaw); near(motion.Bank(), firstBank);
    motion.Update(1.0275, Pi / 2, 1);
    near(motion.Yaw(), Pi / 4);
    motion.Update(1.055, Pi / 2, 1);
    near(motion.Yaw(), Pi / 2);
    motion.Update(1.12, Pi / 2, 1);
    near(motion.Bank(), 0);

    // Sampling at different frame rates follows the same accepted-turn timeline.
    for (int fps : {30, 60, 120, 240})
    {
        State sampled;
        sampled.Update(.99, 0, 0);
        for (int i = 1; double(i) / fps < .05; ++i)
            sampled.Update(1 + double(i) / fps, Pi / 2, 1);
        sampled.Update(1.05, Pi / 2, 1);
        near(sampled.Yaw(), Pi / 2 * Ease(.05 / TurnDuration));
    }

    // Fast reversal remains continuous, bounded, and takes the short path at wrap.
    State reversal;
    reversal.Update(.99, Pi - .1, 0);
    reversal.Update(1.01, -Pi + .1, 1);
    assert(std::abs(Wrap(reversal.Yaw() - (Pi - .1))) < .1);
    reversal.Update(1.04, Pi - .1, 1.04);
    assert(std::isfinite(reversal.Yaw()) && std::abs(reversal.Bank()) <= .25);
    reversal.Update(1.10, Pi - .1, 1.04);
    near(Wrap(reversal.Yaw() - (Pi - .1)), 0);
    reversal.Reset();
    reversal.Update(-3, .75, -4);
    near(reversal.Yaw(), .75); near(reversal.Bank(), 0);
    reversal.Update(-4, -.75, -4); // a new round never animates through the old pose
    near(reversal.Yaw(), -.75); near(reversal.Bank(), 0);
    reversal.Update(std::numeric_limits<double>::quiet_NaN(), 0, 0);
    near(reversal.Yaw(), -.75);

    near(WheelAngle(Pi * .72, .72), 0); // one world-space circumference
    near(WheelAngle(Pi * .72 / 4, .72), Pi / 2);
    near(WheelAngle(12.5, .72), WheelAngle(12.5 + 20 * Pi * .72, .72));
}
