#include "../../../src/tron/gNeuralAsync.h"

#include <atomic>
#include <cassert>
#include <future>
#include <iostream>

using namespace gNeuralAsync;
using Clock = std::chrono::steady_clock;

struct Memory
{
    int state = 0;
    void Reset() { state = 0; }
};

struct ControlledInference
{
    std::mutex mutex;
    std::condition_variable changed;
    bool release = false;
    std::vector< Ticket > calls;
    std::vector< int > priorState;
    std::vector< std::array< float, 7 > > contexts;
    std::thread::id thread;

    int operator ()( Observation const & observation, Memory & memory, uint32_t & rng )
    {
        std::unique_lock< std::mutex > guard( mutex );
        thread = std::this_thread::get_id();
        calls.push_back( observation.ticket );
        priorState.push_back( memory.state );
        contexts.push_back( observation.context );
        ++memory.state;
        ++rng;
        changed.notify_all();
        if ( calls.size() == 1 ) changed.wait( guard, [&] { return release; } );
        // Observations are owned copies; callers may immediately reuse their source arrays.
        assert( observation.maps.front() == uint8_t( observation.ticket.decision ) );
        assert( observation.scalars.front() == float( observation.ticket.decision ) );
        return int( observation.mask == 16 ? 4 : 1 );
    }
    void WaitForCalls( size_t count )
    {
        std::unique_lock< std::mutex > guard( mutex );
        assert( changed.wait_for( guard, std::chrono::seconds( 2 ), [&] { return calls.size() >= count; } ) );
    }
    void Release()
    {
        std::lock_guard< std::mutex > guard( mutex );
        release = true;
        changed.notify_all();
    }
};

Observation Job( uint64_t decision, size_t slot = 0, uint64_t life = 1, uint64_t weights = 1 )
{
    Observation result;
    result.ticket = { weights, life, decision, 1, uint32_t( slot + 10 ), slot };
    result.maps.assign( 64, uint8_t( decision ) );
    result.scalars.assign( 12, float( decision ) );
    result.mask = 63;
    result.gameTime = double( decision ) * .05;
    result.context[2] = 1.0f;
    result.context[6] = 1.0f;
    return result;
}

std::vector< Result > WaitResults( Worker< Memory > & worker, size_t count )
{
    std::vector< Result > results;
    auto deadline = Clock::now() + std::chrono::seconds( 2 );
    while ( results.size() < count && Clock::now() < deadline )
    {
        auto batch = worker.Poll();
        for ( auto & result : batch ) results.push_back( std::move( result ) );
        std::this_thread::yield();
    }
    assert( results.size() == count );
    return results;
}

void TestBoundedQueueAndMemory()
{
    ControlledInference infer;
    Worker< Memory > worker( 2, std::ref( infer ) );
    Observation first = Job( 1 );
    assert( worker.Submit( first ) );
    infer.WaitForCalls( 1 );
    first.maps[0] = 99;
    first.scalars[0] = 99;
    assert( infer.thread != std::this_thread::get_id() );
    // Inference is deliberately blocked. Submission/polling must still complete, and the
    // intermediate job must never run: only the newest pending observation occupies slot 0.
    assert( worker.Submit( Job( 2 ) ) );
    assert( worker.Submit( Job( 3 ) ) );
    assert( !worker.Submit( Job( 2 ) ) );
    assert( !worker.Submit( Job( 3 ) ) );
    assert( worker.Submit( Job( 1, 1 ) ) );
    assert( worker.Poll().empty() );
    infer.Release();
    auto results = WaitResults( worker, 2 );
    assert( results[0].ticket.decision == 3 || results[1].ticket.decision == 3 );
    infer.WaitForCalls( 3 );
    {
        std::lock_guard< std::mutex > guard( infer.mutex );
        assert( infer.calls[0].slot == 0 && infer.calls[0].decision == 1 );
        assert( infer.calls[1].slot == 1 ); // round-robin prevents a busy slot starving another
        assert( infer.calls[2].slot == 0 && infer.calls[2].decision == 3 );
        assert( infer.priorState == std::vector< int >({ 0, 0, 0 }) ); // gap resets old history
        assert( ( infer.contexts[0] == std::array< float, 7 >{} ) ); // birth discards stale control context
        assert( ( infer.contexts[1] == std::array< float, 7 >{} ) ); // another player's independent birth
        assert( infer.contexts[2][2] == 1 && infer.contexts[2][6] == 2 ); // actual skipped elapsed time
    }
    Statistics stats = worker.Stats();
    assert( stats.replaced == 1 && stats.rejected == 2 && stats.stale == 1 );
    assert( stats.gapResets == 1 && stats.skippedDecisions == 1 && stats.lifeResets == 2 );

    assert( worker.Submit( Job( 4 ) ) );
    WaitResults( worker, 1 );
    assert( worker.Submit( Job( 5, 0, 2 ) ) );
    WaitResults( worker, 1 );
    assert( worker.Submit( Job( 6, 0, 2, 2 ) ) );
    WaitResults( worker, 1 );
    {
        std::lock_guard< std::mutex > guard( infer.mutex );
        assert( infer.priorState[3] == 1 ); // consecutive observation retains per-life memory
        assert( infer.priorState[4] == 0 ); // new life resets it
        assert( infer.priorState[5] == 0 ); // new weights reset it
    }
    worker.Stop();
    assert( !worker.Submit( Job( 7 ) ) );
    assert( worker.Poll().empty() );
}

void TestExecutedControlContext()
{
    ControlContext context;
    auto birth = context.Observe( 0 );
    assert( ( birth == std::array< float, 7 >{} ) );
    // A selected but rejected/initial-time-zero action is not an execution.
    context.Applied( -1 );
    auto dropped = context.Observe( .05 );
    for ( int a = 0; a < 6; ++a ) assert( dropped[a] == 0 );
    assert( dropped[6] == 1 );
    context.Applied( 4 );
    auto executed = context.Observe( .125 );
    assert( executed[4] == 1 && executed[6] == 1.5f );
    auto noRepeatedTurn = context.Observe( .175 );
    for ( int a = 0; a < 6; ++a ) assert( noRepeatedTurn[a] == 0 );
    assert( noRepeatedTurn[6] > .999f && noRepeatedTurn[6] < 1.001f );
    context.Reset();
    assert( ( context.Observe( 100 ) == std::array< float, 7 >{} ) );
}

void TestCommitGuards()
{
    Ticket expected = Job( 8 ).ticket;
    Result result;
    result.ticket = expected;
    result.action = 4;
    result.gameTime = 1;
    assert( Fresh( result, 1.04, .05 ) );
    assert( !Fresh( result, 1.06, .05 ) ); // render/main-loop stall made even the newest job too old
    assert( !Fresh( result, .9, .05 ) ); // clock/round rollback may not apply a future result
    assert( CanApply( result, expected, 7, true, 16 ) );
    assert( !CanApply( result, expected, 8, true, 16 ) ); // already applied
    assert( !CanApply( result, expected, 7, false, 16 ) ); // died before completion
    assert( !CanApply( result, expected, 7, true, 1 ) ); // turn legality changed
    for ( int field = 0; field < 6; ++field )
    {
        Ticket other = expected;
        if ( field == 0 ) ++other.weights;
        if ( field == 1 ) ++other.round;
        if ( field == 2 ) ++other.life;
        if ( field == 3 ) ++other.player;
        if ( field == 4 ) ++other.slot;
        if ( field == 5 ) ++other.decision;
        assert( !CanApply( result, other, 0, true, 63 ) );
    }
    result.action = -1;
    assert( !CanApply( result, expected, 0, true, 63 ) );
    result.action = 6;
    assert( !CanApply( result, expected, 0, true, 255 ) );
    result.action = 1;
    result.error = "failed";
    assert( !CanApply( result, expected, 0, true, 63 ) );
}

void TestInvalidationAndJoinedReload()
{
    ControlledInference infer;
    Worker< Memory > worker( 1, std::ref( infer ) );
    worker.Submit( Job( 1 ) );
    infer.WaitForCalls( 1 );
    worker.Invalidate( 0 ); // dead/replaced cycle: its in-flight result may never be published
    worker.Submit( Job( 1, 0, 2 ) );
    infer.Release();
    auto result = WaitResults( worker, 1 );
    assert( result[0].ticket.life == 2 );
    assert( worker.Stats().stale == 1 );
    worker.InvalidateAll();
    assert( worker.Poll().empty() );
    worker.Stop();

    ControlledInference blocked;
    Worker< Memory > reloading( 1, std::ref( blocked ) );
    reloading.Submit( Job( 1 ) );
    blocked.WaitForCalls( 1 );
    std::promise< void > started;
    auto joined = std::async( std::launch::async, [&] { started.set_value(); reloading.Stop(); } );
    started.get_future().wait();
    assert( joined.wait_for( std::chrono::milliseconds( 20 ) ) == std::future_status::timeout );
    blocked.Release();
    assert( joined.wait_for( std::chrono::seconds( 2 ) ) == std::future_status::ready );
    joined.get();
    assert( reloading.Poll().empty() );
    // Only now may the owning game thread replace/load weights and create a fresh worker.
    Worker< Memory > next( 1, []( Observation const &, Memory & memory, uint32_t & ) {
        assert( memory.state == 0 ); return 2;
    } );
    next.Submit( Job( 1, 0, 1, 2 ) );
    assert( WaitResults( next, 1 )[0].action == 2 );
}

void TestFailureDoesNotKillWorker()
{
    unsigned calls = 0;
    Worker< Memory > worker( 1, [&]( Observation const &, Memory & memory, uint32_t & ) {
        if ( calls++ == 0 ) { memory.state = 99; throw std::runtime_error( "test inference failure" ); }
        assert( memory.state == 0 ); return 0;
    } );
    worker.Submit( Job( 1 ) );
    assert( WaitResults( worker, 1 )[0].error == "test inference failure" );
    worker.Submit( Job( 2 ) );
    assert( WaitResults( worker, 1 )[0].action == 0 );
    assert( worker.Stats().failures == 1 );
}

int main()
{
    TestBoundedQueueAndMemory();
    TestCommitGuards();
    TestInvalidationAndJoinedReload();
    TestFailureDoesNotKillWorker();
    TestExecutedControlContext();
    std::cout << "async worker: bounded queue, nonblocking submission, ordering, life/weight resets, "
                 "legality, cancellation, joined reload, and failure recovery passed\n";
}
