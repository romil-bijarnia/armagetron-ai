// Blacklight model animation. Simulation direction and camera state stay separate.
// Licensed under the project's GNU GPL version 2 or later.
#ifndef ARMAGETRON_LIGHTCYCLE_MOTION_H
#define ARMAGETRON_LIGHTCYCLE_MOTION_H

#include <algorithm>
#include <cmath>

namespace gLightcycleMotion
{
constexpr double Pi = 3.14159265358979323846;
constexpr double TurnDuration = .055;

inline double Wrap(double angle)
{
    return std::remainder(angle, 2 * Pi);
}

inline double Ease(double value)
{
    value = std::max(0., std::min(1., value));
    return value * value * (3 - 2 * value);
}

// The rendered model is half its authored size. Distance-based phase avoids the
// old atan(step) approximation and keeps the equal-size wheels synchronized.
inline double WheelAngle(double distance, double modelRadius)
{
    return std::isfinite(distance) && modelRadius > 0
        ? Wrap(2 * distance / modelRadius) : 0;
}

class State
{
public:
    void Reset() { initialized_ = false; }

    void Update(double now, double target, double turnTime)
    {
        if (!std::isfinite(now) || !std::isfinite(target)) return;
        target = Wrap(target);
        if (!initialized_ || now < sampleTime_ || now - sampleTime_ > .5)
        {
            initialized_ = true;
            sampleTime_ = startTime_ = now;
            target_ = startYaw_ = yaw_ = target;
            delta_ = bank_ = startBank_ = peakBank_ = 0;
            return;
        }
        if (std::abs(Wrap(target - target_)) > 1e-6)
        {
            // Start at the accepted turn's time, not the first frame that sees it.
            double start = std::isfinite(turnTime)
                ? std::max(sampleTime_, std::min(now, turnTime)) : now;
            double from = YawAt(start), bank = BankAt(start);
            startTime_ = start;
            startYaw_ = from;
            startBank_ = bank;
            delta_ = Wrap(target - from);
            target_ = target;
            peakBank_ = -std::max(-1., std::min(1., delta_ / (Pi / 2))) * .20;
        }
        yaw_ = Wrap(YawAt(now));
        bank_ = BankAt(now);
        sampleTime_ = now;
    }

    double Yaw() const { return yaw_; }
    double Bank() const { return bank_; }

private:
    double YawAt(double now) const
    {
        return startYaw_ + delta_ * Ease((now - startTime_) / TurnDuration);
    }
    double BankAt(double now) const
    {
        double age = std::max(0., now - startTime_);
        double pulse = Ease(age / .018) * (1 - Ease((age - .018) / .095));
        double bank = startBank_ * (1 - Ease(age / TurnDuration)) + peakBank_ * pulse;
        return std::max(-.25, std::min(.25, bank));
    }
    bool initialized_ = false;
    double sampleTime_ = 0, startTime_ = 0;
    double target_ = 0, startYaw_ = 0, delta_ = 0, yaw_ = 0;
    double startBank_ = 0, peakBank_ = 0, bank_ = 0;
};
}
#endif
