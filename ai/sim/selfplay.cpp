// Self-play with search: see selfplay.h.

#include "selfplay.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace tron
{
namespace
{
inline uint64_t NextU64( uint64_t & s )
{
    // xorshift64*
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

inline float Uniform( uint64_t & s )
{
    return float( ( NextU64( s ) >> 40 ) * ( 1.0 / 16777216.0 ) );
}

inline float Gumbel( uint64_t & s )
{
    float u = Uniform( s );
    if ( u < 1e-9f )
        u = 1e-9f;
    if ( u > 1 - 1e-7f )
        u = 1 - 1e-7f;
    return -std::log( -std::log( u ) );
}


void AppliedContext( World const & before, World const & after, int player, int action,
                     std::array< float, kContext > & context )
{
    context.fill(0.f);
    // The simulator, like gCycle::Act, drops controls while the game timer is zero.
    // Record what was applied even when the preceding sample has no learning target.
    if (before.cycles[player].alive && before.time > 1E-6f)
    {
        int applied = action >= 0 && action < kActions ? action : ACT_STRAIGHT;
        if (!(before.ActionMask(player) >> applied & 1)) applied = Hold(applied);
        context[applied] = 1.f;
    }
    context[kActions] = std::max(0.f, after.time - before.time) / .05f;
}

inline uint8_t Heading( Cycle const & c )
{
    // the frame the minimap and territory maps are drawn in: one of the four cardinal headings
    return c.dx > .5f ? 1 : ( c.dx < -.5f ? 3 : ( c.dy < -.5f ? 2 : 0 ) );
}

void MaskedSoftmax( float const * logits, unsigned mask, float * out, float temperature = 1 )
{
    float best = -1e30f;
    for ( int a = 0; a < kActions; ++a )
        if ( mask >> a & 1 )
            best = std::max( best, logits[a] / temperature );
    float sum = 0;
    for ( int a = 0; a < kActions; ++a )
    {
        out[a] = ( mask >> a & 1 ) ? std::exp( logits[a] / temperature - best ) : 0.f;
        sum += out[a];
    }
    for ( int a = 0; a < kActions; ++a )
        out[a] = sum > 0 ? out[a] / sum : 0.f;
}
}

SelfPlay::SelfPlay( int nGames ): games( nGames )
{
}

uint32_t SelfPlay::Code( int const * actions, int n ) const
{
    uint32_t c = 0;
    for ( int p = n - 1; p >= 0; --p )
        c = c * kActions + uint32_t( actions[p] );
    return c;
}

void SelfPlay::Start( int gi, float size, int n, int const * agents, int const * learn, int rotate, uint64_t seed )
{
    Game & g = games[gi];
    g.n = std::min( n, kMaxPlayers );
    g.size = size;
    for ( int p = 0; p < g.n; ++p )
    {
        g.agents[p] = agents[p];
        g.memory[p].fill(0.f);
        g.context[p].fill(0.f);
        g.generation[p] = 0;
        g.learn[p] = learn[p] != 0;
        g.endDecision[p] = -1;
        g.outcome[p] = 0;
        g.died[p] = false;
        g.havePrev[p] = false;
        g.traj[p].clear();
        g.filled[p] = 0;
        g.hunterAct[p] = ACT_STRAIGHT;
        g.hunterAt[p] = -1000;
        g.rootUpdate[p] = -1;
        g.lastChoice[p] = DecisionWatch();
    }
    g.rng = seed * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
    if ( g.rng == 0 )
        g.rng = 1;
    g.world = World();
    g.world.Reset( size, g.n, rotate );
    g.active = true;
    g.finished = false;
    g.decision = 0;
    g.prevMaps.assign( size_t( g.n ) * kMapBytes, 0 );
    g.curMaps.assign( size_t( g.n ) * kMapBytes, 0 );
    g.curScalars.assign( size_t( g.n ) * World::kScalars, 0 );
    g.searching = false;
    g.nodes.clear();
    g.nodes.reserve( cfg.sims + 2 );
    g.requests.clear();
    g.stage = 0;
    g.waiting = false;
    g.fed = false;
    g.searches = g.sims = 0;
    std::fill( g.sourceCounts, g.sourceCounts + 5, int64_t( 0 ) );
}

void SelfPlay::WatchDecision( Game & g, int p, int action, int selection, float const * search )
{
    DecisionWatch & w = g.lastChoice[p];
    // Retain the observation buffers' capacity between decisions.
    w.maps.clear(); w.previous.clear(); w.scalars.clear();
    w.policyUpdate = -1;
    w.value = 0;
    w.hunterReplanned = false;
    std::fill( w.policy, w.policy + kActions, 0.f );
    std::fill( w.search, w.search + kActions, 0.f );
    w.action = action;
    w.decision = g.decision;
    w.selection = selection;
    w.observedAt = g.world.time;
    w.context = g.context[p];
    ++g.sourceCounts[selection];
    if ( selection == WATCH_HUNTER )
    {
        w.mask = g.world.ActionMask( p );
        w.hunterReplanned = g.hunterAt[p] == g.decision;
        return;
    }
    w.mask = g.world.ActionMask( p );
    w.policyUpdate = g.rootUpdate[p];
    w.value = g.rootValue[p];
    uint8_t const * current = &g.curMaps[size_t( p ) * kMapBytes];
    uint8_t const * previous = g.havePrev[p] ? &g.prevMaps[size_t( p ) * kMapBytes] : current;
    w.maps.assign( current, current + kMapBytes );
    w.previous.assign( previous, previous + kMapBytes );
    float const * scalars = &g.curScalars[size_t( p ) * World::kScalars];
    w.scalars.assign( scalars, scalars + World::kScalars );
    // These are the probabilities the sampler used, or the root network prior for a search
    // or greedy move. The improved search target is a distinct distribution.
    MaskedSoftmax( g.rootLogits[p], w.mask, w.policy,
                   selection == WATCH_SAMPLED ? cfg.temperature : 1.f );
    if ( search )
        std::copy( search, search + kActions, w.search );
}

// ------------------------------------------------------------------ network requests
void SelfPlay::Ask( Game & g, int node, int p, World const & now, World const & before )
{
    auto found = std::find_if(g.requests.begin(), g.requests.end(), [=](Request const & r) {
        return r.node == node && r.player == p;
    });
    if (found == g.requests.end()) {
        g.requests.push_back(Request{node, p});
        found = g.requests.end() - 1;
        found->memory = node < 0 ? g.memory[p] : g.nodes[node].memory[p];
    }
    Request & request = *found;
    size_t offset = request.maps.size(), scalarOffset = request.scalars.size();
    request.maps.resize(offset + 2 * kMapBytes);
    request.scalars.resize(scalarOffset + World::kScalars);
    uint8_t * cur = request.maps.data() + offset;
    uint8_t * prev = cur + kMapBytes;
    int const g2 = World::kGrid * World::kGrid;
    now.Observe(p, cur, cur + g2, cur + 2 * g2, cur + 3 * g2, request.scalars.data() + scalarOffset);
    if (&before == &now)
        std::memcpy(prev, cur, kMapBytes);
    else {
        float dump[World::kScalars];
        before.Observe(p, prev, prev + g2, prev + 2 * g2, prev + 3 * g2, dump);
    }
    request.masks.push_back(uint8_t(now.ActionMask(p)));
    auto const & context = node < 0 ? g.context[p] : g.nodes[node].context[p];
    request.contexts.insert(request.contexts.end(), context.begin(), context.end());
}

// ------------------------------------------------------------------ the real decisions
void SelfPlay::BeginDecision( Game & g )
{
    int const g2 = World::kGrid * World::kGrid;
    g.requests.clear();
    for ( int p = 0; p < g.n; ++p )
    {
        if ( !g.world.cycles[p].alive || g.agents[p] == kAgentHunter )
            continue;
        size_t k = g.requests.size();
        g.requests.push_back( tron::Request{ -1, p } );
        g.reqMaps.resize( ( k + 1 ) * 2 * kMapBytes );
        g.reqScalars.resize( ( k + 1 ) * World::kScalars );
        g.reqMask.resize( k + 1 );
        uint8_t * cur = &g.curMaps[size_t( p ) * kMapBytes];
        float * sc = &g.curScalars[size_t( p ) * World::kScalars];
        g.world.Observe( p, cur, cur + g2, cur + 2 * g2, cur + 3 * g2, sc );
        std::memcpy( &g.reqMaps[k * 2 * kMapBytes], cur, kMapBytes );
        std::memcpy( &g.reqMaps[k * 2 * kMapBytes + kMapBytes], g.havePrev[p] ? &g.prevMaps[size_t( p ) * kMapBytes] : cur, kMapBytes );
        std::memcpy( &g.reqScalars[k * World::kScalars], sc, World::kScalars * sizeof( float ) );
        g.reqMask[k] = uint8_t( g.world.ActionMask( p ) );
        Request & request = g.requests.back();
        request.memory = g.memory[p];
        request.maps.assign(g.reqMaps.begin() + k * 2 * kMapBytes, g.reqMaps.begin() + (k + 1) * 2 * kMapBytes);
        request.scalars.assign(sc, sc + World::kScalars);
        request.masks.push_back(g.reqMask[k]);
        request.contexts.assign(g.context[p].begin(), g.context[p].end());
    }
    g.stage = 1;
}

void SelfPlay::Record( Game & g, int p, bool searched, float const * policy )
{
    if ( !g.learn[p] || g.agents[p] < 0 || g.agents[p] > 3 )
        return;
    Ring & R = rings[g.agents[p]];
    if ( !R.Valid() )
        return;
    int64_t idx = __atomic_fetch_add( R.counter, int64_t( 1 ), __ATOMIC_SEQ_CST );
    int64_t slot = idx % R.cap;
    __atomic_store_n( &R.ready[slot], uint8_t( 0 ), __ATOMIC_RELAXED );
    __atomic_store_n( &R.gen[slot], idx, __ATOMIC_RELEASE );
    uint8_t const * cur = &g.curMaps[size_t( p ) * kMapBytes];
    std::memcpy( R.maps + slot * kMapBytes, cur, kMapBytes );
    std::memcpy( R.scalars + slot * World::kScalars, &g.curScalars[size_t( p ) * World::kScalars], World::kScalars * sizeof( float ) );
    R.mask[slot] = uint8_t( g.world.ActionMask( p ) );
    for ( int a = 0; a < kActions; ++a )
        R.policy[slot * kActions + a] = searched ? policy[a] : 0.f;
    R.hasPolicy[slot] = searched ? 1 : 0;
    R.value[slot] = 0;
    for ( int a = 0; a < kAux; ++a )
        R.aux[slot * kAux + a] = 0;
    if ( R.future )
        R.future[slot] = -1;
    if ( R.heading )
        R.heading[slot] = Heading( g.world.cycles[p] );
    int64_t prev = -1;
    if ( !g.traj[p].empty() && g.traj[p].back().decision == g.decision - 1 )
        prev = g.traj[p].back().index;
    R.prev[slot] = prev;
    if (R.generation) R.generation[slot] = g.generation[p];
    if (R.context) std::copy(g.context[p].begin(), g.context[p].end(), R.context + slot * kContext);
    g.traj[p].push_back( Game::Entry{ idx, g.decision, TerritoryShare( cur + 3 * World::kGrid * World::kGrid ), g.rootValue[p], g.world.cycles[p].kills } );
    statSamples++;
    Fill( g, p, R );
}

// a sample's targets, written only if its slot still holds it; another thread may be reusing the slot
// right now, so the sample is marked ready only if it is still there after the writes
static void WriteTargets( Ring & R, Game::Entry const & e, float value, float share, float dead, float safeKill, int64_t future )
{
    int64_t slot = e.index % R.cap;
    if ( __atomic_load_n( &R.gen[slot], __ATOMIC_ACQUIRE ) != e.index )
        return;  // overwritten already
    R.value[slot] = value;
    R.aux[slot * kAux] = share;
    R.aux[slot * kAux + 1] = dead;
    R.aux[slot * kAux + 2] = safeKill;
    if ( R.future )
        R.future[slot] = future;
    if ( __atomic_load_n( &R.gen[slot], __ATOMIC_ACQUIRE ) == e.index )
        __atomic_store_n( &R.ready[slot], uint8_t( 1 ), __ATOMIC_RELEASE );
}

// n-step targets: once a player is valueSteps decisions past a sample (and alive, or the round would
// have finished it), the sample's value is the discounted value estimate of the newest decision, and
// its aux targets are known (alive auxHorizon decisions later, territory share then)
void SelfPlay::Fill( Game & g, int p, Ring & R )
{
    std::vector< Game::Entry > const & tr = g.traj[p];
    Game::Entry const & now = tr.back();
    int const H = cfg.auxHorizon, N = std::max( cfg.valueSteps, H + 1 );
    while ( g.filled[p] < tr.size() && tr[g.filled[p]].decision + N <= now.decision )
    {
        size_t i = g.filled[p]++;
        // the share auxHorizon decisions later: the entry at that decision (entries are consecutive
        // decisions within a life, so it is i + H; search the few around it if one was skipped)
        size_t j = std::min( i + size_t( H ), tr.size() - 1 );
        while ( j > i && tr[j].decision > tr[i].decision + H )
            --j;
        WriteTargets( R, tr[i], now.value * std::pow( cfg.gamma, float( now.decision - tr[i].decision ) ), tr[j].share, 0.f,
                      tr[j].kills > tr[i].kills ? 1.f : 0.f, tr[j].index );
    }
}

void SelfPlay::MarkDeaths( Game & g, World const & before, World const & after, int decision )
{
    int alive = after.Alive();
    for ( int p = 0; p < g.n; ++p )
        if ( before.cycles[p].alive && !after.cycles[p].alive && g.endDecision[p] < 0 )
        {
            g.endDecision[p] = decision;
            g.outcome[p] = g.n > 1 ? -float( alive ) / float( g.n - 1 ) : -1.f;
            g.died[p] = true;
            FinishPlayer( g, p );
        }
    bool over = after.Over() || decision >= cfg.maxDecisions;
    if ( over )
        for ( int p = 0; p < g.n; ++p )
            if ( after.cycles[p].alive && g.endDecision[p] < 0 )
            {
                g.endDecision[p] = decision;
                g.outcome[p] = ( after.Over() && g.n > 1 ) ? 1.f : 0.f;
                g.died[p] = false;
            }
}

void SelfPlay::FinishDecision( Game & g, int const * acts )
{
    World before = g.world;
    g.world.Step( acts );
    for (int p = 0; p < g.n; ++p)
        AppliedContext(before, g.world, p, acts[p], g.context[p]);
    g.decision++;
    MarkDeaths( g, before, g.world, g.decision );
    for ( int p = 0; p < g.n; ++p )
        if ( before.cycles[p].alive )
        {
            std::memcpy( &g.prevMaps[size_t( p ) * kMapBytes], &g.curMaps[size_t( p ) * kMapBytes], kMapBytes );
            g.havePrev[p] = true;
        }
    for (int p = 0; p < g.n; ++p)
        if (!g.world.cycles[p].alive) {
            g.memory[p].fill(0.f);
            g.context[p].fill(0.f);
        }
    g.searching = false;
    g.stage = 0;
    statDecisions++;
}

// a player's result is known: the samples not yet filled get the discounted outcome (they are the
// last valueSteps decisions of the life). Called at a death, so a long round does not hold them
// back until the ring has overwritten them, and for everyone left when the round ends.
void SelfPlay::FinishPlayer( Game & g, int p )
{
    std::vector< Game::Entry > const & tr = g.traj[p];
    if ( g.filled[p] >= tr.size() )
        return;
    Ring & R = rings[g.agents[p]];
    if ( !R.Valid() )
        return;
    int const H = cfg.auxHorizon;
    int T = g.endDecision[p] >= 0 ? g.endDecision[p] : g.decision;
    float u = g.outcome[p];
    bool died = g.died[p];
    for ( size_t i = g.filled[p]; i < tr.size(); ++i )
    {
        Game::Entry const & e = tr[i];
        float share;
        int64_t future;
        int futureKills;
        if ( i + H < tr.size() )
        {
            share = tr[i + H].share;
            future = tr[i + H].index;
            futureKills = tr[i + H].kills;
        }
        else
        {
            share = died ? 0.f : tr.back().share;
            future = tr.back().index;
            futureKills = g.world.cycles[p].kills;
        }
        // A kill is useful supervision only when the player survives the whole horizon (or
        // survives an earlier terminal round). Posthumous kills and mutual deaths are negative.
        bool deadWithinHorizon = died && T - e.decision <= H;
        WriteTargets( R, e, u * std::pow( cfg.gamma, float( T - e.decision ) ), share,
                      deadWithinHorizon ? 1.f : 0.f,
                      !deadWithinHorizon && futureKills > e.kills ? 1.f : 0.f, future );
    }
    g.filled[p] = tr.size();
}

void SelfPlay::Finish( Game & g )
{
    for ( int p = 0; p < g.n; ++p )
        FinishPlayer( g, p );
    g.finished = true;
    g.active = false;
}

// ------------------------------------------------------------------ search
// A player's completed Q at a node: the mean return of each visited move, the prior-weighted mix
// of the network's value and those returns for the others, then min-max rescaled over the legal
// moves to [0, 1] (mctx's qtransform_completed_by_mix_value). Returns the Gumbel sigma scale
// (cVisit + maxN) * cScale that turns them into logit bonuses. The rescale is what makes the
// search bite: raw returns differ between moves by a few hundredths, and spreading them over the
// full sigma range tells the policy which move the look-ahead preferred instead of adding the
// same bonus to every move.
float CompletedQ( Node const & nd, int p, float cVisit, float cScale, float * q )
{
    float prior[kActions];
    MaskedSoftmax( nd.logits[p], nd.mask[p], prior );
    int sumN = 0, maxN = 0;
    float num = 0, den = 0;
    for ( int a = 0; a < kActions; ++a )
    {
        sumN += nd.N[p][a];
        maxN = std::max( maxN, nd.N[p][a] );
        if ( nd.N[p][a] > 0 )
        {
            num += prior[a] * nd.W[p][a] / nd.N[p][a];
            den += prior[a];
        }
    }
    float vmix = nd.value[p];
    if ( sumN > 0 && den > 0 )
        vmix = ( nd.value[p] + sumN * num / den ) / ( 1 + sumN );
    float lo = 1e30f, hi = -1e30f;
    for ( int a = 0; a < kActions; ++a )
    {
        q[a] = nd.N[p][a] > 0 ? nd.W[p][a] / nd.N[p][a] : vmix;
        if ( nd.mask[p] >> a & 1 )
        {
            lo = std::min( lo, q[a] );
            hi = std::max( hi, q[a] );
        }
    }
    float range = hi - lo;  // all moves equal (or none legal): no preference, every bonus is zero
    for ( int a = 0; a < kActions; ++a )
        q[a] = range > 1e-8f ? ( q[a] - lo ) / range : 0.f;
    return ( cVisit + maxN ) * cScale;
}

void SelfPlay::ImprovedPolicy( Game const & g, Node const & nd, int p, float * out ) const
{
    float q[kActions], z[kActions];
    float scale = CompletedQ( nd, p, cfg.cVisit, cfg.cScale, q );
    for ( int a = 0; a < kActions; ++a )
        z[a] = nd.logits[p][a] + scale * q[a];
    MaskedSoftmax( z, nd.mask[p], out );
    (void)g;
}

float SelfPlay::RootScore( Game & g, int p, int a ) const
{
    Node const & r = g.nodes[0];
    float q[kActions];
    float scale = CompletedQ( r, p, cfg.cVisit, cfg.cScale, q );
    return g.gumbel[p][a] + r.logits[p][a] + scale * q[a];
}

int SelfPlay::SelectRoot( Game & g, int p )
{
    std::vector< int > & c = g.cands[p];
    if ( c.size() == 1 )
        return c[0];
    bool phaseDone = true;
    for ( size_t k = 0; k < c.size(); ++k )
        phaseDone = phaseDone && g.phaseVisits[p][c[k]] >= g.perCand[p];
    if ( phaseDone )
    {
        // sequential halving: keep the better half
        std::sort( c.begin(), c.end(), [&]( int a, int b ) { return RootScore( g, p, a ) > RootScore( g, p, b ); } );
        c.resize( ( c.size() + 1 ) / 2 );
        g.phase[p]++;
        for ( int a = 0; a < kActions; ++a )
            g.phaseVisits[p][a] = 0;
        int left = std::max( 1, g.phases[p] - g.phase[p] );
        g.perCand[p] = std::max( 1, cfg.sims / ( left * int( c.size() ) ) );
        if ( c.size() == 1 )
            return c[0];
    }
    int best = c[0];
    for ( size_t k = 1; k < c.size(); ++k )
        if ( g.phaseVisits[p][c[k]] < g.phaseVisits[p][best] )
            best = c[k];
    g.phaseVisits[p][best]++;
    return best;
}

int SelfPlay::SelectInner( Game const & g, Node const & nd, int p ) const
{
    float pol[kActions];
    ImprovedPolicy( g, nd, p, pol );
    int sumN = 0;
    for ( int a = 0; a < kActions; ++a )
        sumN += nd.N[p][a];
    int best = -1;
    float bestScore = -1e30f;
    for ( int a = 0; a < kActions; ++a )
    {
        if ( !( nd.mask[p] >> a & 1 ) )
            continue;
        float s = pol[a] - float( nd.N[p][a] ) / float( 1 + sumN );
        if ( s > bestScore )
        {
            bestScore = s;
            best = a;
        }
    }
    return best < 0 ? ACT_STRAIGHT : best;
}

void SelfPlay::BeginSearch( Game & g )
{
    g.nodes.clear();
    g.nodes.emplace_back();
    Node & r = g.nodes[0];
    r.w = g.world;
    r.depth = 0;
    r.elapsed = 0;
    r.evaluated = true;
    g.searchUntil = -1;
    for ( int p = 0; p < g.n; ++p )
    {
        bool alive = g.world.cycles[p].alive;
        r.done[p] = !alive;
        r.memory[p] = g.memory[p];
        r.context[p] = g.context[p];
        r.util[p] = g.outcome[p];
        r.value[p] = alive ? g.rootValue[p] : g.outcome[p];
        r.mask[p] = alive ? g.world.ActionMask( p ) : 0;
        r.hunterAct[p] = g.hunterAct[p];
        r.hunterAt[p] = g.hunterAt[p];
        for ( int a = 0; a < kActions; ++a )
        {
            r.logits[p][a] = g.rootLogits[p][a];
            r.N[p][a] = 0;
            r.W[p][a] = 0;
            g.phaseVisits[p][a] = 0;
            g.gumbel[p][a] = cfg.gumbel > 0 ? cfg.gumbel * Gumbel( g.rng ) : 0.f;
        }
        g.cands[p].clear();
        if ( !alive )
            continue;
        if ( g.agents[p] == kAgentHunter )
        {
            int a = HunterAction( r.w, g.decision, p, r.hunterAct[p], r.hunterAt[p] );
            r.mask[p] = 1u << a;
            int next = r.hunterAt[p] + std::max( 1, cfg.hunterReplan );
            g.searchUntil = g.searchUntil < 0 ? next : std::min( g.searchUntil, next );
            continue;
        }
        for ( int a = 0; a < kActions; ++a )
            if ( r.mask[p] >> a & 1 )
                g.cands[p].push_back( a );
        std::sort( g.cands[p].begin(), g.cands[p].end(), [&]( int a, int b ) {
            return g.gumbel[p][a] + r.logits[p][a] > g.gumbel[p][b] + r.logits[p][b]; } );
        int m = int( g.cands[p].size() );
        int ph = 0;
        while ( ( 1 << ph ) < m )
            ++ph;
        g.phases[p] = ph;
        g.phase[p] = 0;
        g.perCand[p] = ph > 0 ? std::max( 1, cfg.sims / ( ph * m ) ) : cfg.sims;
    }
    g.simsDone = 0;
    g.searching = true;
    g.searches++;
    statSearches++;
    g.stage = 2;
}

void SelfPlay::Expand( Game & g, int parent, uint32_t code, int & child )
{
    int acts[kMaxPlayers];
    {
        uint32_t c = code;
        for ( int p = 0; p < g.n; ++p )
        {
            acts[p] = int( c % kActions );
            c /= kActions;
        }
    }
    g.nodes.emplace_back();
    child = int( g.nodes.size() ) - 1;
    Node & nd = g.nodes[child];
    Node & par = g.nodes[parent];
    par.kids.push_back( std::make_pair( code, child ) );
    nd.w = par.w;
    nd.depth = par.depth + 1;
    nd.elapsed = par.elapsed;
    for ( int p = 0; p < g.n; ++p )
    {
        nd.done[p] = par.done[p];
        nd.memory[p] = par.memory[p];
        nd.context[p] = par.context[p];
        nd.util[p] = par.util[p];
        nd.value[p] = par.util[p];
        nd.mask[p] = 0;
        nd.hunterAct[p] = par.hunterAct[p];
        nd.hunterAt[p] = par.hunterAt[p];
        for ( int a = 0; a < kActions; ++a )
        {
            nd.logits[p][a] = 0;
            nd.N[p][a] = 0;
            nd.W[p][a] = 0;
        }
    }
    int baseDecision = g.decision + par.elapsed;
    int duration = cfg.macro;
    if ( g.searchUntil >= 0 )
        duration = std::min( duration, g.searchUntil - baseDecision );
    World before;
    bool over = false;
    g.requests.clear();
    for ( int d = 0; d < duration && !over; ++d )
    {
        int step[kMaxPlayers];
        for ( int p = 0; p < g.n; ++p )
            if ( nd.done[p] )
                step[p] = ACT_STRAIGHT;
            else if ( g.agents[p] == kAgentHunter )
                step[p] = HunterAction( nd.w, baseDecision + d, p, nd.hunterAct[p], nd.hunterAt[p] );
            else
                step[p] = d == 0 ? acts[p] : Hold( acts[p] );
        before = nd.w;
        nd.w.Step( step );
        for (int p = 0; p < g.n; ++p)
            AppliedContext(before, nd.w, p, step[p], nd.context[p]);
        ++nd.elapsed;
        int alive = nd.w.Alive();
        int dec = baseDecision + d + 1;
        for ( int p = 0; p < g.n; ++p )
            if ( !nd.done[p] && !nd.w.cycles[p].alive )
            {
                nd.done[p] = true;
                nd.util[p] = g.n > 1 ? -float( alive ) / float( g.n - 1 ) : -1.f;
            }
        if ( nd.w.Over() || dec >= cfg.maxDecisions )
        {
            for ( int p = 0; p < g.n; ++p )
                if ( !nd.done[p] )
                {
                    nd.done[p] = true;
                    nd.util[p] = ( nd.w.Over() && g.n > 1 ) ? 1.f : 0.f;
                }
            over = true;
        }
        for (int p = 0; p < g.n; ++p)
            if (!nd.done[p] && g.agents[p] != kAgentHunter)
                Ask(g, child, p, nd.w, before);
    }
    // A player that died later along this edge has no continuation to evaluate.
    g.requests.erase(std::remove_if(g.requests.begin(), g.requests.end(), [&](Request const & r) {
        return nd.done[r.player];
    }), g.requests.end());
    nd.cutoff = g.searchUntil >= 0 && g.decision + nd.elapsed >= g.searchUntil;
    bool anyAlive = false;
    for ( int p = 0; p < g.n; ++p )
        anyAlive = anyAlive || !nd.done[p];
    if ( !anyAlive )
    {
        nd.evaluated = true;
        for ( int p = 0; p < g.n; ++p )
            nd.value[p] = nd.util[p];
        return;
    }
    // ask for the network's view of every player still in the round

    for ( int p = 0; p < g.n; ++p )
        if ( !nd.done[p] )
        {
            nd.mask[p] = nd.w.ActionMask( p );
            if ( g.agents[p] == kAgentHunter )
            {
                // Its known control is simulated exactly up to the cutoff. The learner's
                // value bootstraps there; no policy or value is needed for the hunter seat.
                nd.value[p] = 0;
                continue;
            }

        }
    if ( g.requests.empty() )
        nd.evaluated = true;
}

void SelfPlay::Backup( Game & g, int leaf )
{
    float G[kMaxPlayers];
    Node const & lf = g.nodes[leaf];
    for ( int p = 0; p < g.n; ++p )
        G[p] = lf.done[p] ? lf.util[p] : lf.value[p];
    int child = leaf;
    for ( int i = int( g.path.size() ) - 1; i >= 0; --i )
    {
        Node & nd = g.nodes[g.path[i].first];
        float disc = std::pow( cfg.gamma, float( g.nodes[child].elapsed - nd.elapsed ) );
        uint32_t c = g.path[i].second;
        for ( int p = 0; p < g.n; ++p )
        {
            int a = int( c % kActions );
            c /= kActions;
            if ( nd.done[p] )
                continue;
            float ret = disc * G[p];
            nd.N[p][a] += 1;
            nd.W[p][a] += ret;
            G[p] = ret;
        }
        child = g.path[i].first;
    }
}

void SelfPlay::RunSimulations( Game & g )
{
    while ( g.simsDone < cfg.sims )
    {
        g.path.clear();
        int cur = 0;
        while ( true )
        {
            Node & nd = g.nodes[cur];
            bool anyAlive = false;
            for ( int p = 0; p < g.n; ++p )
                anyAlive = anyAlive || !nd.done[p];
            if ( !anyAlive || nd.cutoff )
            {
                Backup( g, cur );
                g.simsDone++;
                g.sims++;
                statSims++;
                break;
            }
            int acts[kMaxPlayers];
            for ( int p = 0; p < g.n; ++p )
                if ( nd.done[p] )
                    acts[p] = ACT_STRAIGHT;
                else if ( g.agents[p] == kAgentHunter )
                {
                    acts[p] = HunterAction( nd.w, g.decision + nd.elapsed, p,
                                            nd.hunterAct[p], nd.hunterAt[p] );
                    nd.mask[p] = 1u << acts[p];
                }
                else
                    acts[p] = cur == 0 ? SelectRoot( g, p ) : SelectInner( g, nd, p );
            uint32_t code = Code( acts, g.n );
            int next = -1;
            for ( size_t k = 0; k < nd.kids.size(); ++k )
                if ( nd.kids[k].first == code )
                {
                    next = nd.kids[k].second;
                    break;
                }
            g.path.push_back( std::make_pair( cur, code ) );
            if ( next >= 0 )
            {
                cur = next;
                continue;
            }
            int child;
            Expand( g, cur, code, child );
            if ( g.nodes[child].evaluated )
            {
                Backup( g, child );
                g.simsDone++;
                g.sims++;
                statSims++;
                break;
            }
            g.pendingNode = child;
            g.stage = 3;
            return;  // wait for the network
        }
    }

    // the search is over: improved policies, moves, samples
    int acts[kMaxPlayers];
    for ( int p = 0; p < g.n; ++p )
    {
        acts[p] = ACT_STRAIGHT;
        if ( !g.world.cycles[p].alive )
            continue;
        if ( g.agents[p] == kAgentHunter )
        {
            acts[p] = HunterAction( g, p );
            WatchDecision( g, p, acts[p], WATCH_HUNTER );
            continue;
        }
        float pol[kActions];
        ImprovedPolicy( g, g.nodes[0], p, pol );
        std::vector< int > & c = g.cands[p];
        int best = c.empty() ? ACT_STRAIGHT : c[0];
        for ( size_t k = 1; k < c.size(); ++k )
            if ( RootScore( g, p, c[k] ) > RootScore( g, p, best ) )
                best = c[k];
        acts[p] = best;
        Record( g, p, true, pol );
        WatchDecision( g, p, acts[p], WATCH_SEARCH, pol );
    }
    FinishDecision( g, acts );
}

// ------------------------------------------------------------------ the space fighter
// The search itself lives in fighter.cpp, shared with the game's bots.
static FighterConfig Fighter( SPConfig const & cfg )
{
    FighterConfig f;
    f.depth = cfg.hunterDepth;
    f.ply = cfg.hunterPly;
    f.probe = cfg.hunterProbe;
    return f;
}

float SelfPlay::HunterEval( World const & w, int p, int e ) const
{
    return FighterEval( w, p, e, Fighter( cfg ) );
}

int SelfPlay::HunterMove( World const & w0, int p ) const
{
    return FighterMove( w0, p, Fighter( cfg ) );
}

int SelfPlay::HunterAction( Game & g, int p )
{
    return HunterAction( g.world, g.decision, p, g.hunterAct[p], g.hunterAt[p] );
}

int SelfPlay::HunterAction( World const & w, int decision, int p, int & action, int & plannedAt ) const
{
    int hold = std::max( 1, cfg.hunterReplan );
    if ( decision >= plannedAt && decision < plannedAt + hold )
        return decision == plannedAt ? action : Hold( action );
    action = HunterMove( w, p );
    plannedAt = decision;
    return action;
}

// the real decision, once the network has answered (or nobody needed it)
void SelfPlay::Decide( Game & g )
{
    bool anyNet = false;
    for ( int p = 0; p < g.n; ++p )
    {
        if ( !g.world.cycles[p].alive )
            continue;
        if ( g.agents[p] != kAgentHunter )
        {
            anyNet = true;
            continue;
        }
        // the space fighter has decided; a search at this decision sees its move as all but certain
        int a = HunterAction( g, p );
        for ( int b = 0; b < kActions; ++b )
            g.rootLogits[p][b] = b == a ? 8.f : -8.f;
        g.rootValue[p] = 0;
    }
    bool search = anyNet && g.decision > 0 && cfg.searchProb > 0 && cfg.sims > 0 && Uniform( g.rng ) < cfg.searchProb;
    if ( search )
    {
        BeginSearch( g );
        return;
    }
    int acts[kMaxPlayers];
    for ( int p = 0; p < g.n; ++p )
    {
        acts[p] = ACT_STRAIGHT;
        if ( !g.world.cycles[p].alive )
            continue;
        if ( g.agents[p] == kAgentHunter )
        {
            acts[p] = HunterAction( g, p );
            WatchDecision( g, p, acts[p], WATCH_HUNTER );
            continue;
        }
        unsigned mask = g.world.ActionMask( p );
        float pol[kActions];
        if ( cfg.temperature <= 0 )
        {
            int best = ACT_STRAIGHT;
            float bl = -1e30f;
            for ( int a = 0; a < kActions; ++a )
                if ( ( mask >> a & 1 ) && g.rootLogits[p][a] > bl )
                {
                    bl = g.rootLogits[p][a];
                    best = a;
                }
            acts[p] = best;
        }
        else
        {
            MaskedSoftmax( g.rootLogits[p], mask, pol, cfg.temperature );
            float r = Uniform( g.rng ), acc = 0;
            acts[p] = ACT_STRAIGHT;
            for ( int a = 0; a < kActions; ++a )
            {
                acc += pol[a];
                if ( r < acc && ( mask >> a & 1 ) )
                {
                    acts[p] = a;
                    break;
                }
            }
        }
        Record( g, p, false, 0 );
        WatchDecision( g, p, acts[p], cfg.temperature <= 0 ? WATCH_GREEDY : WATCH_SAMPLED );
    }
    FinishDecision( g, acts );
}

// ------------------------------------------------------------------ driving a round
void SelfPlay::Advance( Game & g )
{
    if ( !g.active || g.finished )
        return;
    if ( g.waiting && !g.fed )
        return;
    if ( g.fed )
    {
        g.fed = false;
        g.waiting = false;
        if ( g.stage == 1 )
        {
            for ( size_t k = 0; k < g.requests.size(); ++k )
            {
                int p = g.requests[k].player;
                for ( int a = 0; a < kActions; ++a )
                    g.rootLogits[p][a] = g.outLogits[k * kActions + a];
                g.rootValue[p] = g.outValue[k];
                g.rootUpdate[p] = g.outUpdates[k];
                std::copy_n(g.outMemory.data() + k * kMemory, kMemory, g.memory[p].begin());
            }
            g.requests.clear();
            Decide( g );
        }
        else if ( g.stage == 3 )
        {
            Node & nd = g.nodes[g.pendingNode];
            for ( size_t k = 0; k < g.requests.size(); ++k )
            {
                int p = g.requests[k].player;
                for ( int a = 0; a < kActions; ++a )
                    nd.logits[p][a] = g.outLogits[k * kActions + a];
                nd.value[p] = g.outValue[k];
                std::copy_n(g.outMemory.data() + k * kMemory, kMemory, nd.memory[p].begin());
            }
            nd.evaluated = true;
            g.requests.clear();
            Backup( g, g.pendingNode );
            g.simsDone++;
            g.sims++;
            statSims++;
            g.stage = 2;
        }
    }
    while ( true )
    {
        if ( g.stage == 0 )
        {
            if ( g.world.Over() || g.decision >= cfg.maxDecisions )
            {
                Finish( g );
                return;
            }
            BeginDecision( g );
            if ( g.requests.empty() )
            {
                Decide( g );  // nobody needs the network: the space fighters move on their own
                continue;
            }
            g.waiting = true;
            return;
        }
        if ( g.stage == 2 )
        {
            RunSimulations( g );
            if ( g.stage == 3 )
            {
                g.waiting = true;
                return;
            }
            continue;  // decided: next decision
        }
        return;
    }
}

int SelfPlay::Collect( uint8_t * maps, float * scalars, uint8_t * masks, int * agentIds, int * gameIds, int maxBatch,
                       uint8_t * roots, float * memory, int * lengths,
                       uint8_t * sequenceMaps, float * sequenceScalars, uint8_t * sequenceMasks,
                       float * sequenceContexts )
{
    int const nGames = int( games.size() );
    int nThreads = std::max( 1, std::min( cfg.threads, nGames ) );
    if ( nThreads == 1 )
        for ( int i = 0; i < nGames; ++i )
            Advance( games[i] );
    else
    {
        std::atomic< int > next( 0 );
        std::vector< std::thread > pool;
        for ( int t = 0; t < nThreads; ++t )
            pool.emplace_back( [&]() {
                for ( int i = next++; i < nGames; i = next++ )
                    Advance( games[i] );
            } );
        for ( size_t t = 0; t < pool.size(); ++t )
            pool[t].join();
    }
    batchGames_.clear();
    int b = 0;
    for ( int i = 0; i < nGames; ++i )
    {
        Game & g = games[i];
        if ( !g.waiting || g.fed )
            continue;
        if ( b + int( g.requests.size() ) > maxBatch )
            break;
        g.outLogits.assign( g.requests.size() * kActions, 0.f );
        g.outValue.assign( g.requests.size(), 0.f );
        g.outUpdates.assign( g.requests.size(), -1 );
        g.outMemory.assign(g.requests.size() * kMemory, 0.f);
        for ( size_t k = 0; k < g.requests.size(); ++k, ++b )
        {
            Request const & request = g.requests[k];
            size_t count = request.masks.size(), last = count - 1;
            std::memcpy(maps + size_t(b) * 2 * kMapBytes, request.maps.data() + last * 2 * kMapBytes, 2 * kMapBytes);
            std::memcpy(scalars + size_t(b) * World::kScalars, request.scalars.data() + last * World::kScalars, World::kScalars * sizeof(float));
            if (memory) std::copy(request.memory.begin(), request.memory.end(), memory + size_t(b) * kMemory);
            if (lengths) lengths[b] = int(count);
            size_t base = size_t(b) * std::max(cfg.macro, 1);
            if (sequenceMaps) std::copy(request.maps.begin(), request.maps.end(), sequenceMaps + base * 2 * kMapBytes);
            if (sequenceScalars) std::copy(request.scalars.begin(), request.scalars.end(), sequenceScalars + base * World::kScalars);
            if (sequenceMasks) std::copy(request.masks.begin(), request.masks.end(), sequenceMasks + base);
            if (sequenceContexts) std::copy(request.contexts.begin(), request.contexts.end(), sequenceContexts + base * kContext);
            masks[b] = request.masks.back();
            agentIds[b] = g.agents[g.requests[k].player];
            gameIds[b] = i;
            if ( roots )
                roots[b] = g.requests[k].node < 0 ? 1 : 0;
            batchGames_.push_back( i );
        }
    }
    return b;
}

void SelfPlay::Feed( float const * logits, float const * values, int batch, int64_t const * updates, float const * memory )
{
    int b = 0;
    while ( b < batch && b < int( batchGames_.size() ) )
    {
        Game & g = games[batchGames_[b]];
        for ( size_t k = 0; k < g.requests.size(); ++k, ++b )
        {
            for ( int a = 0; a < kActions; ++a )
                g.outLogits[k * kActions + a] = logits[size_t( b ) * kActions + a];
            g.outValue[k] = values[b];
            g.outUpdates[k] = updates ? updates[b] : -1;
            if (memory) std::copy_n(memory + size_t(b) * kMemory, kMemory, g.outMemory.data() + k * kMemory);
        }
        g.fed = true;
    }
}
}
