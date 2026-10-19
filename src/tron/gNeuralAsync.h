#ifndef ARMAGETRON_NEURAL_ASYNC_H
#define ARMAGETRON_NEURAL_ASYNC_H

// Local policy inference only. Jobs/results contain plain copied observations and identifiers;
// the worker must never access game objects, configuration globals, SDL, or OpenGL. There is one
// in-flight inference and at most one queued observation/result per configured player slot.
// New observations replace queued ones. Memory/RNG belong to the worker and advance only on
// processed observations; a changed life/weights or skipped decision resets Memory before use.
// This trades inference plus up to one main-loop tick of action latency for an unblocked render
// loop. The main thread must validate the complete ticket, life and current legality again.

#include <chrono>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gNeuralAsync
{
struct Ticket
{
    uint64_t weights = 0, life = 0, decision = 0;
    uint32_t round = 0, player = 0;
    size_t slot = 0;

    bool SameLife( Ticket const & other ) const
    {
        return weights == other.weights && round == other.round && player == other.player &&
               slot == other.slot && life == other.life;
    }
    bool operator ==( Ticket const & other ) const
    {
        return SameLife( other ) && decision == other.decision;
    }
    bool NewerThan( Ticket const & other ) const
    {
        if ( slot != other.slot ) return false;
        if ( weights != other.weights ) return weights > other.weights;
        if ( round != other.round ) return round > other.round;
        if ( life != other.life ) return life > other.life;
        return SameLife( other ) && decision > other.decision;
    }
};

struct Observation
{
    Ticket ticket;
    std::vector< uint8_t > maps;
    std::vector< float > scalars;
    unsigned mask = 0;
    bool sample = false;
    double gameTime = 0;
    std::array< float, 7 > context{}; // last executed action, then elapsed game time / 0.05
    std::chrono::steady_clock::time_point submitted;
};

// Main-thread control history. Selecting a move must never call Applied: only a
// control accepted by the final life/age/legality checks is an executed action.
struct ControlContext
{
    int action = -1;
    bool haveObservation = false;
    double observedAt = 0;
    void Reset() { action = -1; haveObservation = false; observedAt = 0; }
    void Applied( int executed ) { if ( executed >= 0 && executed < 6 ) action = executed; }
    std::array< float, 7 > Observe( double time )
    {
        std::array< float, 7 > result{};
        if ( haveObservation )
        {
            if ( action >= 0 ) result[size_t( action )] = 1.0f;
            result[6] = float( std::max( 0.0, time - observedAt ) / 0.05 );
        }
        observedAt = time;
        haveObservation = true;
        action = -1; // no newly applied control means no repeated turn event next interval
        return result;
    }
};

struct Result
{
    Ticket ticket;
    int action = -1;
    double gameTime = 0, computeMs = 0, latencyMs = 0;
    std::chrono::steady_clock::time_point submitted;
    std::string error;
};

inline bool Fresh( Result const & result, double gameTime, double decisionInterval )
{
    return gameTime + 1e-5 >= result.gameTime && gameTime - result.gameTime <= decisionInterval + 1e-5;
}

inline bool CanApply( Result const & result, Ticket const & expected, uint64_t lastApplied,
                      bool alive, unsigned legalMask )
{
    return alive && result.error.empty() && result.ticket == expected &&
           result.ticket.decision > lastApplied && result.action >= 0 && result.action < 6 &&
           ( legalMask & ( 1u << result.action ) );
}

struct Statistics
{
    uint64_t submitted = 0, replaced = 0, rejected = 0, processed = 0, stale = 0;
    uint64_t lifeResets = 0, gapResets = 0, skippedDecisions = 0, failures = 0;
    double computeMs = 0, latencyMs = 0;
};

template< class Memory > class Worker
{
public:
    using Infer = std::function< int( Observation const &, Memory &, uint32_t & ) >;

    Worker( size_t capacity, Infer infer ) : queues_( capacity ), states_( capacity ), infer_( std::move( infer ) )
    {
        if ( capacity == 0 ) throw std::invalid_argument( "local inference needs at least one slot" );
        thread_ = std::thread( &Worker::Run, this );
    }
    ~Worker() { Stop(); }
    Worker( Worker const & ) = delete;
    Worker & operator =( Worker const & ) = delete;

    size_t Capacity() const { return queues_.size(); }

    bool Submit( Observation observation )
    {
        std::lock_guard< std::mutex > guard( mutex_ );
        if ( stopping_ || observation.ticket.slot >= queues_.size() ) return false;
        Queue & queue = queues_[observation.ticket.slot];
        if ( queue.valid && !observation.ticket.NewerThan( queue.expected ) )
        {
            ++statistics_.rejected;
            return false;
        }
        if ( queue.pending ) ++statistics_.replaced;
        if ( queue.ready ) ++statistics_.stale;
        observation.submitted = std::chrono::steady_clock::now();
        queue.expected = observation.ticket;
        queue.observation = std::move( observation );
        queue.valid = queue.pending = true;
        queue.ready = false;
        ++statistics_.submitted;
        wake_.notify_one();
        return true;
    }

    // Short queue operations only: no inference, game object access, or joins while holding it.
    std::vector< Result > Poll()
    {
        std::lock_guard< std::mutex > guard( mutex_ );
        std::vector< Result > results;
        for ( Queue & queue : queues_ )
            if ( queue.ready )
            {
                results.push_back( std::move( queue.result ) );
                queue.ready = false;
            }
        return results;
    }

    void Invalidate( size_t slot )
    {
        std::lock_guard< std::mutex > guard( mutex_ );
        if ( slot < queues_.size() ) Clear( queues_[slot] );
    }
    void InvalidateAll()
    {
        std::lock_guard< std::mutex > guard( mutex_ );
        for ( Queue & queue : queues_ ) Clear( queue );
    }
    Statistics Stats() const
    {
        std::lock_guard< std::mutex > guard( mutex_ );
        return statistics_;
    }
    void Stop()
    {
        {
            std::lock_guard< std::mutex > guard( mutex_ );
            stopping_ = true;
            for ( Queue & queue : queues_ ) Clear( queue );
        }
        wake_.notify_one();
        if ( thread_.joinable() ) thread_.join();
    }

private:
    struct Queue
    {
        bool valid = false, pending = false, ready = false;
        Ticket expected;
        Observation observation;
        Result result;
    };
    struct State
    {
        bool have = false;
        Ticket last;
        Memory memory;
        uint32_t rng = 1;
        double observedAt = 0;
    };
    std::vector< Queue > queues_;
    std::vector< State > states_; // worker-only; extensible to recurrent Memory
    Infer infer_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    size_t nextSlot_ = 0;
    Statistics statistics_;
    std::thread thread_;

    void Clear( Queue & queue )
    {
        if ( queue.ready ) ++statistics_.stale;
        queue.valid = queue.pending = queue.ready = false;
    }
    bool Pending() const
    {
        for ( Queue const & queue : queues_ ) if ( queue.pending ) return true;
        return false;
    }
    void Run()
    {
        for (;;)
        {
            Observation observation;
            {
                std::unique_lock< std::mutex > guard( mutex_ );
                wake_.wait( guard, [&] { return stopping_ || Pending(); } );
                if ( stopping_ ) return;
                for ( size_t offset = 0; offset < queues_.size(); ++offset )
                {
                    size_t slot = ( nextSlot_ + offset ) % queues_.size();
                    Queue & queue = queues_[slot];
                    if ( !queue.pending ) continue;
                    observation = std::move( queue.observation );
                    queue.pending = false;
                    nextSlot_ = ( slot + 1 ) % queues_.size();
                    break;
                }
            }

            State & state = states_[observation.ticket.slot];
            bool lifeReset = !state.have || !state.last.SameLife( observation.ticket );
            bool gapReset = !lifeReset && observation.ticket.decision != state.last.decision + 1;
            uint64_t skipped = gapReset && observation.ticket.decision > state.last.decision ?
                               observation.ticket.decision - state.last.decision - 1 : 0;
            if ( lifeReset || gapReset ) state.memory.Reset();
            if ( lifeReset )
            {
                observation.context.fill( 0.0f );
                state.rng = 2463534242u ^ uint32_t( observation.ticket.life ) ^
                            ( observation.ticket.player * 2654435761u ) ^ uint32_t( observation.ticket.weights );
                if ( !state.rng ) state.rng = 1;
            }
            else
                observation.context[6] = float( std::max( 0.0, observation.gameTime - state.observedAt ) / 0.05 );
            Result result;
            result.ticket = observation.ticket;
            result.gameTime = observation.gameTime;
            result.submitted = observation.submitted;
            auto start = std::chrono::steady_clock::now();
            try { result.action = infer_( observation, state.memory, state.rng ); }
            catch ( std::exception const & error ) { result.error = error.what(); }
            catch (...) { result.error = "unknown local inference error"; }
            auto end = std::chrono::steady_clock::now();
            result.computeMs = std::chrono::duration< double, std::milli >( end - start ).count();
            result.latencyMs = std::chrono::duration< double, std::milli >( end - observation.submitted ).count();
            state.last = observation.ticket;
            state.observedAt = observation.gameTime;
            state.have = result.error.empty();

            std::lock_guard< std::mutex > guard( mutex_ );
            ++statistics_.processed;
            statistics_.lifeResets += lifeReset;
            statistics_.gapResets += gapReset;
            statistics_.skippedDecisions += skipped;
            statistics_.failures += !result.error.empty();
            statistics_.computeMs += result.computeMs;
            statistics_.latencyMs += result.latencyMs;
            if ( stopping_ ) return;
            Queue & queue = queues_[observation.ticket.slot];
            if ( queue.valid && queue.expected == result.ticket )
            {
                queue.result = std::move( result );
                queue.ready = true;
            }
            else ++statistics_.stale;
        }
    }
};
}
#endif
