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

inline int Hold( int a )
{
    // inside a tree edge a turn is followed by driving straight; braking keeps braking
    return a == ACT_BRAKE ? ACT_BRAKE : ACT_STRAIGHT;
}

float TerritoryShare( uint8_t const * territory )
{
    int mine = 0, theirs = 0;
    for ( int i = 0; i < World::kGrid * World::kGrid; ++i )
    {
        mine += territory[i] & 1;
        theirs += ( territory[i] >> 1 ) & 1;
    }
    return mine + theirs > 0 ? float( mine ) / float( mine + theirs ) : 0.f;
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
        g.learn[p] = learn[p] != 0;
        g.endDecision[p] = -1;
        g.outcome[p] = 0;
        g.died[p] = false;
        g.havePrev[p] = false;
        g.traj[p].clear();
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
}

// ------------------------------------------------------------------ network requests
void SelfPlay::Ask( Game & g, int node, int p, World const & now, World const & before )
{
    size_t k = g.requests.size();
    g.requests.push_back( tron::Request{ node, p } );
    g.reqMaps.resize( ( k + 1 ) * 2 * kMapBytes );
    g.reqScalars.resize( ( k + 1 ) * World::kScalars );
    g.reqMask.resize( k + 1 );
    uint8_t * cur = &g.reqMaps[k * 2 * kMapBytes];
    uint8_t * prev = cur + kMapBytes;
    int const g2 = World::kGrid * World::kGrid;
    float * sc = &g.reqScalars[k * World::kScalars];
    now.Observe( p, cur, cur + g2, cur + 2 * g2, cur + 3 * g2, sc );
    if ( &before == &now )
        std::memcpy( prev, cur, kMapBytes );
    else
    {
        static thread_local std::vector< float > dump( World::kScalars );
        before.Observe( p, prev, prev + g2, prev + 2 * g2, prev + 3 * g2, &dump[0] );
    }
    g.reqMask[k] = uint8_t( now.ActionMask( p ) );
}

// ------------------------------------------------------------------ the real decisions
void SelfPlay::BeginDecision( Game & g )
{
    int const g2 = World::kGrid * World::kGrid;
    g.requests.clear();
    for ( int p = 0; p < g.n; ++p )
    {
        if ( !g.world.cycles[p].alive )
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
    R.ready[slot] = 0;
    R.gen[slot] = idx;
    uint8_t const * cur = &g.curMaps[size_t( p ) * kMapBytes];
    std::memcpy( R.maps + slot * kMapBytes, cur, kMapBytes );
    std::memcpy( R.scalars + slot * World::kScalars, &g.curScalars[size_t( p ) * World::kScalars], World::kScalars * sizeof( float ) );
    R.mask[slot] = uint8_t( g.world.ActionMask( p ) );
    for ( int a = 0; a < kActions; ++a )
        R.policy[slot * kActions + a] = searched ? policy[a] : 0.f;
    R.hasPolicy[slot] = searched ? 1 : 0;
    R.value[slot] = 0;
    R.aux[slot * 2] = R.aux[slot * 2 + 1] = 0;
    int64_t prev = -1;
    if ( !g.traj[p].empty() && g.traj[p].back().decision == g.decision - 1 )
        prev = g.traj[p].back().index;
    R.prev[slot] = prev;
    g.traj[p].push_back( Game::Entry{ idx, g.decision, TerritoryShare( cur + 3 * World::kGrid * World::kGrid ) } );
    statSamples++;
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
    g.decision++;
    MarkDeaths( g, before, g.world, g.decision );
    for ( int p = 0; p < g.n; ++p )
        if ( before.cycles[p].alive )
        {
            std::memcpy( &g.prevMaps[size_t( p ) * kMapBytes], &g.curMaps[size_t( p ) * kMapBytes], kMapBytes );
            g.havePrev[p] = true;
        }
    g.searching = false;
    g.stage = 0;
    statDecisions++;
}

void SelfPlay::Finish( Game & g )
{
    int const H = cfg.auxHorizon;
    for ( int p = 0; p < g.n; ++p )
    {
        if ( g.traj[p].empty() )
            continue;
        Ring & R = rings[g.agents[p]];
        if ( !R.Valid() )
            continue;
        int T = g.endDecision[p] >= 0 ? g.endDecision[p] : g.decision;
        float u = g.outcome[p];
        bool died = g.died[p];
        std::vector< Game::Entry > const & tr = g.traj[p];
        for ( size_t i = 0; i < tr.size(); ++i )
        {
            Game::Entry const & e = tr[i];
            int64_t slot = e.index % R.cap;
            if ( R.gen[slot] != e.index )
                continue;  // overwritten already
            R.value[slot] = u * std::pow( cfg.gamma, float( T - e.decision ) );
            float share;
            if ( i + H < tr.size() )
                share = tr[i + H].share;
            else
                share = died ? 0.f : tr.back().share;
            R.aux[slot * 2] = share;
            R.aux[slot * 2 + 1] = ( died && T - e.decision <= H ) ? 1.f : 0.f;
            R.ready[slot] = 1;
        }
    }
    g.finished = true;
    g.active = false;
}

// ------------------------------------------------------------------ search
void SelfPlay::ImprovedPolicy( Game const & g, Node const & nd, int p, float * out ) const
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
    float scale = ( cfg.cVisit + maxN ) * cfg.cScale;
    float z[kActions];
    for ( int a = 0; a < kActions; ++a )
    {
        float q = nd.N[p][a] > 0 ? nd.W[p][a] / nd.N[p][a] : vmix;
        z[a] = nd.logits[p][a] + scale * ( q + 1 ) * .5f;
    }
    MaskedSoftmax( z, nd.mask[p], out );
    (void)g;
}

float SelfPlay::RootScore( Game & g, int p, int a ) const
{
    Node const & r = g.nodes[0];
    float pol[kActions];
    // completed Q through the improved policy's ingredients
    float prior[kActions];
    MaskedSoftmax( r.logits[p], r.mask[p], prior );
    int sumN = 0, maxN = 0;
    float num = 0, den = 0;
    for ( int b = 0; b < kActions; ++b )
    {
        sumN += r.N[p][b];
        maxN = std::max( maxN, r.N[p][b] );
        if ( r.N[p][b] > 0 )
        {
            num += prior[b] * r.W[p][b] / r.N[p][b];
            den += prior[b];
        }
    }
    float vmix = r.value[p];
    if ( sumN > 0 && den > 0 )
        vmix = ( r.value[p] + sumN * num / den ) / ( 1 + sumN );
    float q = r.N[p][a] > 0 ? r.W[p][a] / r.N[p][a] : vmix;
    (void)pol;
    return g.gumbel[p][a] + r.logits[p][a] + ( cfg.cVisit + maxN ) * cfg.cScale * ( q + 1 ) * .5f;
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
    r.evaluated = true;
    for ( int p = 0; p < g.n; ++p )
    {
        bool alive = g.world.cycles[p].alive;
        r.done[p] = !alive;
        r.util[p] = g.outcome[p];
        r.value[p] = alive ? g.rootValue[p] : g.outcome[p];
        r.mask[p] = alive ? g.world.ActionMask( p ) : 0;
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
    for ( int p = 0; p < g.n; ++p )
    {
        nd.done[p] = par.done[p];
        nd.util[p] = par.util[p];
        nd.value[p] = par.util[p];
        nd.mask[p] = 0;
        for ( int a = 0; a < kActions; ++a )
        {
            nd.logits[p][a] = 0;
            nd.N[p][a] = 0;
            nd.W[p][a] = 0;
        }
    }
    int baseDecision = g.decision + par.depth * cfg.macro;
    World before;
    bool over = false;
    for ( int d = 0; d < cfg.macro && !over; ++d )
    {
        int step[kMaxPlayers];
        for ( int p = 0; p < g.n; ++p )
            step[p] = d == 0 ? acts[p] : Hold( acts[p] );
        before = nd.w;
        nd.w.Step( step );
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
    }
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
    g.requests.clear();
    World const & prevWorld = cfg.macro == 1 ? par.w : before;
    for ( int p = 0; p < g.n; ++p )
        if ( !nd.done[p] )
        {
            Ask( g, child, p, nd.w, prevWorld );
            nd.mask[p] = nd.w.ActionMask( p );
        }
}

void SelfPlay::Backup( Game & g, int leaf )
{
    float G[kMaxPlayers];
    Node const & lf = g.nodes[leaf];
    for ( int p = 0; p < g.n; ++p )
        G[p] = lf.done[p] ? lf.util[p] : lf.value[p];
    float disc = std::pow( cfg.gamma, float( cfg.macro ) );
    for ( int i = int( g.path.size() ) - 1; i >= 0; --i )
    {
        Node & nd = g.nodes[g.path[i].first];
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
            if ( !anyAlive )
            {
                Backup( g, cur );
                g.simsDone++;
                g.sims++;
                statSims++;
                break;
            }
            int acts[kMaxPlayers];
            for ( int p = 0; p < g.n; ++p )
                acts[p] = nd.done[p] ? ACT_STRAIGHT : ( cur == 0 ? SelectRoot( g, p ) : SelectInner( g, nd, p ) );
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
        float pol[kActions];
        ImprovedPolicy( g, g.nodes[0], p, pol );
        std::vector< int > & c = g.cands[p];
        int best = c.empty() ? ACT_STRAIGHT : c[0];
        for ( size_t k = 1; k < c.size(); ++k )
            if ( RootScore( g, p, c[k] ) > RootScore( g, p, best ) )
                best = c[k];
        acts[p] = best;
        Record( g, p, true, pol );
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
            }
            g.requests.clear();
            bool search = g.decision > 0 && cfg.searchProb > 0 && cfg.sims > 0 && Uniform( g.rng ) < cfg.searchProb;
            if ( search )
                BeginSearch( g );
            else
            {
                int acts[kMaxPlayers];
                for ( int p = 0; p < g.n; ++p )
                {
                    acts[p] = ACT_STRAIGHT;
                    if ( !g.world.cycles[p].alive )
                        continue;
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
                }
                FinishDecision( g, acts );
            }
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

int SelfPlay::Collect( uint8_t * maps, float * scalars, uint8_t * masks, int * agentIds, int * gameIds, int maxBatch )
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
        for ( size_t k = 0; k < g.requests.size(); ++k, ++b )
        {
            std::memcpy( maps + size_t( b ) * 2 * kMapBytes, &g.reqMaps[k * 2 * kMapBytes], 2 * kMapBytes );
            std::memcpy( scalars + size_t( b ) * World::kScalars, &g.reqScalars[k * World::kScalars], World::kScalars * sizeof( float ) );
            masks[b] = g.reqMask[k];
            agentIds[b] = g.agents[g.requests[k].player];
            gameIds[b] = i;
            batchGames_.push_back( i );
        }
    }
    return b;
}

void SelfPlay::Feed( float const * logits, float const * values, int batch )
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
        }
        g.fed = true;
    }
}
}
