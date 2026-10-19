/* Map live event timestamps onto the game clock without introducing a new clock. */
#ifndef ARMAGETRON_INPUT_TIME_H
#define ARMAGETRON_INPUT_TIME_H

#include <algorithm>
#include <cmath>
#include <cstdint>

class uInputTimeMapper
{
public:
    void Reset()
    {
        havePrevious_ = false;
        interpolate_ = false;
    }

    // The SDL tick snapshot must correspond to the frame whose game time is supplied.
    // Interpolating the two game times also accounts for pause, speed and TIME_FACTOR.
    void BeginFrame( std::uint32_t ticks, double time, double minimum, bool continuous = true )
    {
        bool finite = std::isfinite(time) && std::isfinite(minimum);
        time_ = std::isfinite(time) ? time : 0;
        minimum_ = std::min(time_, std::isfinite(minimum) ? minimum : time_);
        ticks_ = ticks;
        tickSpan_ = ticks - previousTicks_;
        startTime_ = previousTime_;
        interpolate_ = continuous && finite && havePrevious_ && time >= previousTime_
            && tickSpan_ > 0 && tickSpan_ < 0x80000000u;
        lastEventTime_ = interpolate_ ? std::max(minimum_, startTime_) : time_;

        havePrevious_ = continuous && finite;
        previousTicks_ = ticks;
        previousTime_ = time_;
    }

    double CurrentTime() const { return time_; }

    double EventTime( std::uint32_t eventTicks )
    {
        double result = time_;
        std::uint32_t age = ticks_ - eventTicks;
        // Zero means no timestamp. A negative/wrap-ambiguous age is a future event.
        if ( interpolate_ && eventTicks != 0 && age < 0x80000000u )
        {
            double fraction = 1.0 - double(std::min(age, tickSpan_)) / tickSpan_;
            result = startTime_ + fraction * (time_ - startTime_);
        }
        // The queue's order wins over a stale or out-of-order timestamp. Never rewind.
        lastEventTime_ = std::max(lastEventTime_, std::max(minimum_, std::min(time_, result)));
        return lastEventTime_;
    }

private:
    bool havePrevious_ = false;
    bool interpolate_ = false;
    std::uint32_t previousTicks_ = 0, ticks_ = 0, tickSpan_ = 0;
    double previousTime_ = 0, startTime_ = 0, time_ = 0, minimum_ = 0, lastEventTime_ = 0;
};

#endif
