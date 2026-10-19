// Replay-target and search-macro regressions through SelfPlay's public driving API.
#include "../../sim/selfplay.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

using namespace tron;

struct Fixture
{
    static int const cap = 128;
    SelfPlay sp{ 1 };
    std::vector< uint8_t > maps, masks, hasPolicy, ready, headings, requestMaps, requestMasks;
    std::vector< float > scalars, policy, value, aux, requestScalars, contexts;
    std::vector< int64_t > prev, gen, future;
    std::vector< int > agents, games;
    int64_t counter = 0;
    int batch = 0;

    explicit Fixture( int n = 3 )
        : maps( cap * kMapBytes ), masks( cap ), hasPolicy( cap ), ready( cap ), headings( cap ),
          requestMaps( kMaxPlayers * 2 * kMapBytes ), requestMasks( kMaxPlayers ),
          scalars( cap * World::kScalars ), policy( cap * kActions ), value( cap ), aux( cap * kAux ),
          requestScalars( kMaxPlayers * World::kScalars ), contexts( cap * kContext ),
          prev( cap, -1 ), gen( cap, -1 ), future( cap, -1 ),
          agents( kMaxPlayers ), games( kMaxPlayers )
    {
        Ring & r = sp.rings[0];
        r.maps = maps.data(); r.scalars = scalars.data(); r.mask = masks.data();
        r.policy = policy.data(); r.hasPolicy = hasPolicy.data(); r.value = value.data(); r.aux = aux.data();
        r.prev = prev.data(); r.gen = gen.data(); r.ready = ready.data(); r.counter = &counter;
        r.future = future.data(); r.heading = headings.data(); r.cap = cap;
        r.context = contexts.data();
        sp.cfg.sims = 0;
        sp.cfg.searchProb = 0;
        sp.cfg.temperature = 0;
        sp.cfg.threads = 1;
        sp.cfg.auxHorizon = 2;
        sp.cfg.valueSteps = 3;
        sp.cfg.maxDecisions = 8;
        int ids[kMaxPlayers] = {}, learn[kMaxPlayers] = { 1 };
        sp.Start( 0, 0, n, ids, learn, 0, 5 );
        sp.games[0].world.s.rubber = 0;
        Collect();
    }

    void Collect()
    {
        batch = sp.Collect( requestMaps.data(), requestScalars.data(), requestMasks.data(),
                            agents.data(), games.data(), kMaxPlayers );
    }

    void Next( int action = ACT_STRAIGHT )
    {
        assert( batch > 0 );
        std::vector< float > logits( batch * kActions, -100 );
        for ( int i = 0; i < batch; ++i )
            logits[i * kActions + ( i == 0 ? action : ACT_STRAIGHT )] = 100;
        Answer( logits );
    }

    void Answer( std::vector< float > const & logits )
    {
        assert( logits.size() == size_t( batch * kActions ) );
        std::vector< float > values( batch, 0 );
        sp.Feed( logits.data(), values.data(), batch );
        Collect();
    }

    void RimDeath( int p, int creditedTo = -1 )
    {
        World & w = sp.games[0].world;
        // The last influencing enemy owns the credit, even if the final collision is the rim.
        w.Place( p, w.maxX - .01f, 30.f + 20.f * p, 1, 0 );
        w.cycles[p].hunter = creditedTo;
    }

    void Finish()
    {
        for ( int guard = 0; batch > 0 && guard < 32; ++guard )
            Next();
        assert( sp.games[0].finished );
        for ( int i = 0; i < counter; ++i )
            assert( ready[i] == 1 );
    }
};

static void MacroKeepsBraking( int action )
{
    Fixture f( 2 );
    f.Next();  // actions at time zero are ignored
    f.sp.cfg.sims = 1;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    f.sp.cfg.macro = 3;
    World expected = f.sp.games[0].world;
    int acts[] = { action, ACT_STRAIGHT };
    expected.Step( acts );
    acts[0] = ACT_BRAKE;
    expected.Step( acts );
    expected.Step( acts );
    f.Next( action );  // now waiting for the first searched leaf
    assert( f.sp.games[0].nodes.size() == 2 );
    Cycle const & got = f.sp.games[0].nodes[1].w.cycles[0];
    Cycle const & want = expected.cycles[0];
    assert( got.braking && got.dx == want.dx && got.dy == want.dy );
    assert( std::fabs( got.Speed() - want.Speed() ) < 1e-5f );
    assert( std::fabs( got.brakingReservoir - want.brakingReservoir ) < 1e-5f );
}

static void SearchLegalActions( bool lowFinite = false, bool turnBlocked = false )
{
    Fixture f( 2 );
    f.Next();
    f.sp.cfg.sims = 24;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    Game & game = f.sp.games[0];
    if ( turnBlocked )
    {
        for ( int p = 0; p < game.n; ++p )
            game.world.cycles[p].lastTurnLeft = game.world.time;
        game.requests.clear(); game.waiting = false; game.fed = false; game.stage = 0;
        f.Collect();
    }
    int const decision = game.decision;
    for ( int guard = 0; game.decision == decision && guard < 64; ++guard )
    {
        std::vector< float > logits( f.batch * kActions, 0 );
        for ( int i = 0; i < f.batch; ++i )
        {
            // Any finite logit remains a possible action, even an extremely unlikely one.
            logits[i * kActions + ACT_LEFT_BRAKE] = lowFinite ? -1e8f : 20.f;
            logits[i * kActions + ACT_RIGHT_BRAKE] = lowFinite ? -1e9f : 19.f;
        }
        f.Answer( logits );
    }
    assert( game.decision == decision + 1 );
    bool innerVisited = false;
    for ( Node const & node : game.nodes )
    {
        innerVisited = innerVisited || node.depth >= 2;
        for ( int p = 0; p < game.n; ++p )
            if ( !node.done[p] )
                assert( node.mask[p] == node.w.ActionMask( p ) );
        // Every hypothetical joint move must obey the turn cooldown of its own world.
        for ( auto const & edge : node.kids )
        {
            uint32_t code = edge.first;
            for ( int p = 0; p < game.n; ++p, code /= kActions )
                assert( node.w.ActionMask( p ) & ( 1u << ( code % kActions ) ) );
        }
    }
    assert( innerVisited );
    if ( turnBlocked )
    {
        assert( game.nodes[0].mask[0] == ( ( 1u << ACT_STRAIGHT ) | ( 1u << ACT_BRAKE ) ) );
        for ( int a : { ACT_LEFT, ACT_RIGHT, ACT_LEFT_BRAKE, ACT_RIGHT_BRAKE } )
        {
            assert( game.nodes[0].N[0][a] == 0 && f.policy[kActions + a] == 0 );
            assert( !( f.masks[1] & ( 1u << a ) ) );
        }
        assert( game.world.cycles[0].dx == 0 );
    }
    else
    {
        assert( game.nodes[0].N[0][ACT_LEFT_BRAKE] > 0 && game.nodes[0].N[0][ACT_RIGHT_BRAKE] > 0 );
        assert( ( f.masks[1] & 0b110000 ) == 0b110000 );
        if ( !lowFinite )
            assert( game.world.cycles[0].braking && game.world.cycles[0].dx == -1 );
    }
}

static void HunterBranchesMatchKnownControl( int macro, int replan, int cachedAction )
{
    Fixture f( 2 );
    f.Next();
    Game & game = f.sp.games[0];
    game.agents[1] = kAgentHunter;
    game.hunterAct[1] = cachedAction;
    game.hunterAt[1] = 0;
    f.sp.cfg.sims = 24;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    f.sp.cfg.macro = macro;
    f.sp.cfg.hunterReplan = replan;
    f.sp.cfg.maxDecisions = 100;
    int const decision = game.decision;
    // Refresh requests after changing the opponent from a network seat to the hunter.
    game.requests.clear(); game.waiting = false; game.fed = false; game.stage = 0;
    f.Collect();
    for ( int guard = 0; game.decision == decision && guard < 64; ++guard )
        f.Answer( std::vector< float >( f.batch * kActions, 0 ) );
    assert( game.decision == decision + 1 );
    // The search must stop before a replan and preserve the real round's cached control.
    assert( game.hunterAct[1] == cachedAction && game.hunterAt[1] == 0 );
    int const rootAction = cachedAction >= ACT_BRAKE ? ACT_BRAKE : ACT_STRAIGHT;
    assert( game.lastChoice[1].action == rootAction );
    assert( game.searchUntil == replan );
    bool reachedCutoff = false;
    for ( Node const & parent : game.nodes )
        for ( auto const & edge : parent.kids )
        {
            Node const & child = game.nodes[edge.second];
            reachedCutoff = reachedCutoff || child.cutoff;
            assert( decision + child.elapsed <= replan );
            assert( child.elapsed - parent.elapsed <= macro );
            assert( child.hunterAct[1] == cachedAction && child.hunterAt[1] == 0 );
            assert( !child.cutoff || child.kids.empty() );
            int learnerAction = edge.first % kActions;
            if ( parent.depth == 0 )
                assert( ( edge.first / kActions ) % kActions == unsigned( rootAction ) );
            // Drive the same branch through ordinary, unsearched decisions. This invokes the
            // real hunter controller at every 0.05 s decision up to the next replan.
            Fixture reference( 2 );
            reference.sp.cfg = f.sp.cfg;
            reference.sp.cfg.sims = 0;
            reference.sp.cfg.searchProb = 0;
            Game & live = reference.sp.games[0];
            live.world = parent.w;
            live.decision = decision + parent.elapsed;
            live.agents[1] = kAgentHunter;
            live.learn[0] = false;
            live.hunterAct[1] = parent.hunterAct[1];
            live.hunterAt[1] = parent.hunterAt[1];
            live.requests.clear(); live.waiting = false; live.fed = false; live.stage = 0;
            reference.Collect();
            for ( int d = 0; d < child.elapsed - parent.elapsed && !live.finished; ++d )
            {
                int held = learnerAction >= ACT_BRAKE ? ACT_BRAKE : ACT_STRAIGHT;
                reference.Next( d == 0 ? learnerAction : held );
            }
            assert( std::fabs( child.w.time - live.world.time ) < 1e-6f );
            for ( int p = 0; p < 2; ++p )
            {
                Cycle const & got = child.w.cycles[p], & want = live.world.cycles[p];
                assert( got.alive == want.alive && got.braking == want.braking );
                assert( got.dx == want.dx && got.dy == want.dy );
                assert( std::fabs( got.x - want.x ) < 1e-5f && std::fabs( got.y - want.y ) < 1e-5f );
                assert( std::fabs( got.Speed() - want.Speed() ) < 1e-5f );
                assert( std::fabs( got.brakingReservoir - want.brakingReservoir ) < 1e-5f );
            }
            assert( live.hunterAt[1] == 0 && live.hunterAct[1] == cachedAction );
        }
    assert( reachedCutoff );
}

static void HunterShortEdgeUsesActualDiscountAndHistory()
{
    Fixture f( 2 );
    f.Next();
    Game & game = f.sp.games[0];
    game.agents[1] = kAgentHunter;
    game.hunterAct[1] = ACT_STRAIGHT;
    game.hunterAt[1] = 0;  // at decision 1, just one known control remains before the replan at 2
    f.sp.cfg.sims = 12;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    f.sp.cfg.macro = 3;
    f.sp.cfg.gamma = .5f;
    f.sp.cfg.maxDecisions = 100;
    int const decision = game.decision;
    std::vector< uint8_t > rootMaps( kMapBytes );
    std::vector< float > scalars( World::kScalars );
    int const g2 = World::kGrid * World::kGrid;
    game.world.Observe( 0, rootMaps.data(), rootMaps.data() + g2, rootMaps.data() + 2 * g2,
                        rootMaps.data() + 3 * g2, scalars.data() );
    game.requests.clear(); game.waiting = false; game.fed = false; game.stage = 0;
    f.Collect();
    int leafRequests = 0;
    for ( int guard = 0; game.decision == decision && guard < 64; ++guard )
    {
        if ( game.stage == 3 )
        {
            ++leafRequests;
            assert( f.batch == 1 && game.nodes[game.pendingNode].elapsed == 1 );
            assert( game.nodes[game.pendingNode].cutoff );
            assert( std::equal( rootMaps.begin(), rootMaps.end(), f.requestMaps.begin() + kMapBytes ) );
        }
        std::vector< float > logits( f.batch * kActions, 0 ), values( f.batch, .75f );
        f.sp.Feed( logits.data(), values.data(), f.batch );
        f.Collect();
    }
    assert( game.decision == decision + 1 && leafRequests == kActions );
    Node const & root = game.nodes[0];
    for ( int a = 0; a < kActions; ++a )
    {
        assert( root.N[0][a] > 0 );
        assert( std::fabs( root.W[0][a] / root.N[0][a] - .375f ) < 1e-6f );
    }
}

static void NetworkOnlySearchKeepsFullDepth()
{
    Fixture f( 2 );
    f.Next();
    Game & game = f.sp.games[0];
    f.sp.cfg.sims = 24;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    f.sp.cfg.macro = 3;
    f.sp.cfg.maxDecisions = 100;
    int const decision = game.decision;
    for ( int guard = 0; game.decision == decision && guard < 64; ++guard )
        f.Answer( std::vector< float >( f.batch * kActions, 0 ) );
    bool inner = false;
    assert( game.searchUntil == -1 && game.decision == decision + 1 );
    for ( Node const & node : game.nodes )
    {
        inner = inner || node.depth >= 2;
        assert( !node.cutoff && node.elapsed == node.depth * f.sp.cfg.macro );
    }
    assert( inner );
}

static void HunterCutIsNotAvoidedByAnImaginaryTurn()
{
    Fixture f( 2 );
    f.Next();
    Game & game = f.sp.games[0];
    game.world.Place( 0, 100.f, 100.f, 0, 1 );
    game.world.Place( 1, 98.2f, 98.2f, 0, 1 );
    game.agents[1] = kAgentHunter;
    // This straight plan is held for the two decisions of the searched edge.
    game.hunterAct[1] = ACT_STRAIGHT;
    game.hunterAt[1] = game.decision;
    f.sp.cfg.sims = 24;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    f.sp.cfg.maxDecisions = 100;
    int const decision = game.decision;
    World actual = game.world, imaginary = game.world;
    int actualMoves[] = { ACT_LEFT, ACT_STRAIGHT }, imaginaryMoves[] = { ACT_LEFT, ACT_LEFT };
    actual.Step( actualMoves ); imaginary.Step( imaginaryMoves );
    actualMoves[0] = imaginaryMoves[0] = imaginaryMoves[1] = ACT_STRAIGHT;
    actual.Step( actualMoves ); imaginary.Step( imaginaryMoves );
    assert( !actual.cycles[0].alive && imaginary.cycles[0].alive );
    game.requests.clear(); game.waiting = false; game.fed = false; game.stage = 0;
    f.Collect();
    for ( int guard = 0; game.decision == decision && guard < 64; ++guard )
        f.Answer( std::vector< float >( f.batch * kActions, 0 ) );
    bool checked = false;
    for ( auto const & edge : game.nodes[0].kids )
        if ( edge.first % kActions == ACT_LEFT )
        {
            checked = true;
            assert( ( edge.first / kActions ) % kActions == ACT_STRAIGHT );
            assert( !game.nodes[edge.second].w.cycles[0].alive );
        }
    assert( checked );
}

static void RecurrentBranchContextIsPrivate()
{
    Fixture f(2);
    Game & game = f.sp.games[0];
    for (auto const & request : game.requests)
        assert(std::all_of(request.contexts.begin(), request.contexts.end(), [](float x) { return x == 0; }));
    f.Next(ACT_LEFT_BRAKE); // time-zero input is dropped, rather than reported as an applied turn
    assert(game.context[0][ACT_LEFT_BRAKE] == 0 && std::abs(game.context[0][kActions] - 1) < 1E-4f);
    f.sp.cfg.sims = 2;
    f.sp.cfg.searchProb = 1;
    f.sp.cfg.gumbel = 0;
    f.sp.cfg.macro = 2;
    int const rootDecision = game.decision;
    float const rootTime = game.world.time;
    std::vector<float> logits(f.batch * kActions, -100), values(f.batch, 0), state(f.batch * kMemory, 7.f);
    for (int i = 0; i < f.batch; ++i) logits[i * kActions + ACT_LEFT_BRAKE] = 100;
    f.sp.Feed(logits.data(), values.data(), f.batch, nullptr, state.data());
    f.Collect();
    bool checkedMacro = false;
    for (int guard = 0; game.decision == rootDecision && guard < 128; ++guard)
    {
        assert(game.stage == 3 && game.world.time == rootTime);
        assert(game.memory[0][0] == 7.f && game.context[0][ACT_LEFT_BRAKE] == 0);
        for (auto const & request : game.requests)
        {
            assert(request.contexts.size() == request.masks.size() * kContext);
            for (size_t step = 0; step < request.masks.size(); ++step)
            {
                float const * context = request.contexts.data() + step * kContext;
                float sum = 0;
                for (int a = 0; a < kActions; ++a) sum += context[a];
                assert(sum == 1 && std::abs(context[kActions] - 1) < 1E-4f);
            }
            if (request.masks.size() == 2 && request.contexts[ACT_LEFT_BRAKE] == 1)
            {
                checkedMacro = true;
                assert(request.contexts[kContext + ACT_BRAKE] == 1); // held brake, not a second turn
            }
        }
        std::vector<float> branchLogits(f.batch * kActions, 0), branchValues(f.batch, 0), branchState(f.batch * kMemory, 99.f);
        f.sp.Feed(branchLogits.data(), branchValues.data(), f.batch, nullptr, branchState.data());
        f.Collect();
    }
    assert(checkedMacro && game.decision == rootDecision + 1);
    assert(game.memory[0][0] == 7.f); // no hypothetical state is copied back into the real agent
    assert(game.requests[0].memory[0] == 7.f);
    assert(game.context[0][ACT_LEFT_BRAKE] == 1);
    assert(f.contexts[kContext + kActions] == 1); // context is captured before targets are ready
    assert(f.ready[1] == 0);
}

int main()
{
    RecurrentBranchContextIsPrivate();
    MacroKeepsBraking( ACT_LEFT_BRAKE );
    MacroKeepsBraking( ACT_RIGHT_BRAKE );
    SearchLegalActions();
    SearchLegalActions( true );
    SearchLegalActions( false, true );
    HunterBranchesMatchKnownControl( 1, 3, ACT_STRAIGHT );
    HunterBranchesMatchKnownControl( 2, 2, ACT_LEFT_BRAKE );
    HunterBranchesMatchKnownControl( 3, 3, ACT_STRAIGHT );
    HunterShortEdgeUsesActualDiscountAndHistory();
    NetworkOnlySearchKeepsFullDepth();
    HunterCutIsNotAvoidedByAnImaginaryTurn();
    {
        // A credited kill in the horizon produces an early target while the round continues.
        Fixture f;
        f.Next();
        f.RimDeath( 1, 0 );
        f.Next();
        f.Next();
        f.Next();
        assert( !f.sp.games[0].finished && f.ready[0] == 1 );
        assert( f.aux[1] == 0 && f.aux[2] == 1 );
        f.Finish();
        assert( f.aux[2 * kAux + 2] == 0 ); // kills before this sample do not count again
    }
    {
        Fixture f;
        f.Next();
        f.RimDeath( 1 ); // an enemy's uncredited rim crash is not my kill
        f.Next();
        f.Finish();
        assert( f.aux[2] == 0 );
    }
    {
        Fixture f;
        f.Next();
        f.RimDeath( 0, 1 );
        f.RimDeath( 1, 0 );
        f.Next();
        f.Finish();
        assert( f.sp.games[0].world.cycles[0].kills == 1 );
        assert( f.aux[1] == 1 && f.aux[2] == 0 ); // a credited mutual kill is not safe
    }
    {
        Fixture f( 2 );
        f.Next();
        f.RimDeath( 1, 0 );
        f.Next();
        f.Finish();
        assert( f.aux[1] == 0 && f.aux[2] == 1 ); // surviving a terminal win before H counts
    }
    {
        Fixture f;
        f.sp.cfg.valueSteps = 60; // force targets through FinishPlayer, not the early Fill path
        f.RimDeath( 1, 0 );
        f.Next();
        f.Next();
        f.Next();
        f.RimDeath( 0 );
        f.Next();
        f.Finish();
        assert( f.aux[1] == 0 && f.aux[2] == 1 ); // death after H does not erase a safe-horizon kill
    }
}
