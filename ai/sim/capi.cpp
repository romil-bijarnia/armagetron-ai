// C entry points for Python (ctypes): one World per handle.

#include "tron.h"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

using tron::World;

extern "C"
{
void * tr_new()
{
    return new World();
}

void tr_free( void * w )
{
    delete static_cast< World * >( w );
}

void tr_copy( void * dst, void const * src )
{
    *static_cast< World * >( dst ) = *static_cast< World const * >( src );
}

void tr_reset( void * w, float sizeFactor, int n, int rotate, int const * teams )
{
    static_cast< World * >( w )->Reset( sizeFactor, n, rotate, teams );
}

void tr_set_bounds( void * w, float minX, float minY, float maxX, float maxY )
{
    World * x = static_cast< World * >( w );
    x->minX = minX;
    x->minY = minY;
    x->maxX = maxX;
    x->maxY = maxY;
}

void tr_place( void * w, int i, float x, float y, float dx, float dy )
{
    static_cast< World * >( w )->Place( i, x, y, dx, dy );
}

void tr_step( void * w, int const * actions )
{
    static_cast< World * >( w )->Step( actions );
}

void tr_act( void * w, int i, int action )
{
    static_cast< World * >( w )->Act( i, action );
}

void tr_frame( void * w )
{
    static_cast< World * >( w )->Frame();
}

unsigned tr_mask( void * w, int i )
{
    World * x = static_cast< World * >( w );
    return x->cycles[i].alive ? x->ActionMask( i ) : 0;
}

void tr_observe( void * w, int i, uint8_t * maps, float * scalars )
{
    int const g2 = World::kGrid * World::kGrid;
    static_cast< World * >( w )->Observe( i, maps, maps + g2, maps + 2 * g2, maps + 3 * g2, scalars );
}

//! per cycle: x, y, dx, dy, speed, alive, rubber, braking reservoir, braking, distance, kills, team
int tr_state( void * w, float * out )
{
    World * x = static_cast< World * >( w );
    int n = int( x->cycles.size() );
    for ( int i = 0; i < n; ++i )
    {
        tron::Cycle const & c = x->cycles[i];
        float * o = out + 12 * i;
        o[0] = c.x;
        o[1] = c.y;
        o[2] = c.dx;
        o[3] = c.dy;
        o[4] = c.Speed();
        o[5] = c.alive ? 1 : 0;
        o[6] = c.rubber;
        o[7] = c.brakingReservoir;
        o[8] = c.braking ? 1 : 0;
        o[9] = c.distance;
        o[10] = float( c.kills );
        o[11] = float( c.team );
    }
    return n;
}

float tr_time( void * w )
{
    return static_cast< World * >( w )->time;
}

int tr_over( void * w )
{
    return static_cast< World * >( w )->Over() ? 1 : 0;
}

void tr_bounds( void * w, float * out )
{
    World * x = static_cast< World * >( w );
    out[0] = x->minX;
    out[1] = x->minY;
    out[2] = x->maxX;
    out[3] = x->maxY;
}
}

// ------------------------------------------------------------------ self-play with search
#include "selfplay.h"

using tron::SelfPlay;

namespace
{
// Draw only walls that are solid at the world's current time, including finite trail ends
// and holes made by explosions. This reads the real world; search nodes never appear here.
void WatchSegments( std::ostream & out, World const & w, int p )
{
    tron::Cycle const & c = w.cycles[p];
    bool first = true;
    out << '[';
    for ( size_t i = 0; i < c.trail.size(); ++i )
    {
        tron::Coord const & a = c.trail[i];
        float x = c.x, y = c.y, d = c.distance;
        if ( i + 1 < c.trail.size() )
        {
            x = c.trail[i + 1].x;
            y = c.trail[i + 1].y;
            d = c.trail[i + 1].dist;
        }
        if ( d <= a.dist )
            continue;
        float end = c.distance;
        if ( c.alive )
            end += c.rubberSpeedFactor * c.Speed() * ( w.time - c.lastTime );
        float lo = a.dist, hi = std::min( d, end );
        if ( w.s.wallsLength > 0 )
            lo = std::max( lo, end - w.s.wallsLength );
        if ( hi <= lo )
            continue;
        std::vector< float > cuts{ lo, hi };
        for ( auto const & hole : c.holes )
        {
            if ( hole.first > lo && hole.first < hi ) cuts.push_back( hole.first );
            if ( hole.second > lo && hole.second < hi ) cuts.push_back( hole.second );
        }
        std::sort( cuts.begin(), cuts.end() );
        for ( size_t j = 1; j < cuts.size(); ++j )
        {
            if ( cuts[j] <= cuts[j - 1] || !w.WallPointDangerous( p, ( cuts[j] + cuts[j - 1] ) * .5f, w.time ) )
                continue;
            float t0 = ( cuts[j - 1] - a.dist ) / ( d - a.dist );
            float t1 = ( cuts[j] - a.dist ) / ( d - a.dist );
            if ( !first ) out << ',';
            first = false;
            out << '[' << a.x + t0 * ( x - a.x ) << ',' << a.y + t0 * ( y - a.y ) << ','
                << a.x + t1 * ( x - a.x ) << ',' << a.y + t1 * ( y - a.y ) << ']';
        }
    }
    out << ']';
}

void WatchProbabilities( std::ostream & out, float const * probabilities )
{
    out << '[';
    for ( int a = 0; a < tron::kActions; ++a )
    {
        if ( a ) out << ',';
        if ( std::isfinite( probabilities[a] ) ) out << probabilities[a];
        else out << "null";
    }
    out << ']';
}

void WatchBytes( std::ostream & out, std::vector< uint8_t > const & bytes )
{
    static char const alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    out << '"';
    for ( size_t i = 0; i < bytes.size(); i += 3 )
    {
        uint32_t v = uint32_t( bytes[i] ) << 16;
        if ( i + 1 < bytes.size() ) v |= uint32_t( bytes[i + 1] ) << 8;
        if ( i + 2 < bytes.size() ) v |= bytes[i + 2];
        out << alphabet[( v >> 18 ) & 63] << alphabet[( v >> 12 ) & 63]
            << ( i + 1 < bytes.size() ? alphabet[( v >> 6 ) & 63] : '=' )
            << ( i + 2 < bytes.size() ? alphabet[v & 63] : '=' );
    }
    out << '"';
}
}

extern "C"
{
void * sp_new( int games )
{
    return new SelfPlay( games );
}

void sp_free( void * h )
{
    delete static_cast< SelfPlay * >( h );
}

void sp_config( void * h, int sims, int macro, float gamma, float cVisit, float cScale, float searchProb,
                float temperature, float gumbel, int maxDecisions, int auxHorizon, int threads, int valueSteps,
                int hunterDepth, int hunterReplan, int hunterPly, int hunterProbe )
{
    tron::SPConfig & c = static_cast< SelfPlay * >( h )->cfg;
    c.hunterDepth = hunterDepth;
    c.hunterReplan = hunterReplan;
    c.hunterPly = hunterPly < 1 ? 1 : hunterPly;
    c.hunterProbe = hunterProbe;
    c.sims = sims;
    c.macro = macro < 1 ? 1 : macro;
    c.gamma = gamma;
    c.cVisit = cVisit;
    c.cScale = cScale;
    c.searchProb = searchProb;
    c.temperature = temperature;
    c.gumbel = gumbel;
    c.maxDecisions = maxDecisions;
    c.auxHorizon = auxHorizon;
    c.threads = threads;
    c.valueSteps = valueSteps;
}

void sp_set_ring( void * h, int agent, uint8_t * maps, float * scalars, uint8_t * mask, float * policy, uint8_t * hasPolicy,
                  float * value, float * aux, int64_t * prev, int64_t * gen, uint8_t * ready, int64_t * counter, int64_t cap,
                  int64_t * future, uint8_t * heading, int64_t * generation, float * context )
{
    if ( agent < 0 || agent > 3 )
        return;
    tron::Ring & r = static_cast< SelfPlay * >( h )->rings[agent];
    r.maps = maps;
    r.scalars = scalars;
    r.mask = mask;
    r.policy = policy;
    r.hasPolicy = hasPolicy;
    r.value = value;
    r.aux = aux;
    r.prev = prev;
    r.gen = gen;
    r.ready = ready;
    r.counter = counter;
    r.cap = cap;
    r.future = future;
    r.heading = heading;
    r.generation = generation;
    r.context = context;
}

void sp_start( void * h, int g, float size, int n, int const * agents, int const * learn, int rotate, unsigned long long seed )
{
    static_cast< SelfPlay * >( h )->Start( g, size, n, agents, learn, rotate, seed );
}

void sp_set_generations(void * h, int gi, int64_t const * generations) {
    auto & game = static_cast<SelfPlay *>(h)->games.at(gi);
    std::copy_n(generations, game.n, game.generation);
}

// A curriculum may change fresh spawn positions only, before observations or actions exist.
int sp_place_start( void * h, int gi, int n, float const * poses )
{
    SelfPlay & sp = *static_cast< SelfPlay * >( h );
    if ( gi < 0 || gi >= int( sp.games.size() ) )
        return 0;
    tron::Game & g = sp.games[gi];
    if ( !g.active || g.decision != 0 || g.stage != 0 || g.waiting || g.fed || n != g.n )
        return 0;
    for ( int p = 0; p < n; ++p )
    {
        float const * s = poses + 4 * p;
        for ( int k = 0; k < 4; ++k )
            if ( !std::isfinite( s[k] ) )
                return 0;
        if ( s[0] <= g.world.minX || s[0] >= g.world.maxX || s[1] <= g.world.minY || s[1] >= g.world.maxY ||
             !( ( std::fabs( s[2] ) == 1 && s[3] == 0 ) || ( s[2] == 0 && std::fabs( s[3] ) == 1 ) ) )
            return 0;
        for ( int q = 0; q < p; ++q )
            if ( std::hypot( s[0] - poses[4 * q], s[1] - poses[4 * q + 1] ) < 1 )
                return 0;
    }
    for ( int p = 0; p < n; ++p )
        g.world.Place( p, poses[4 * p], poses[4 * p + 1], poses[4 * p + 2], poses[4 * p + 3] );
    return 1;
}

int sp_collect( void * h, uint8_t * maps, float * scalars, uint8_t * masks, int * agents, int * games, int maxBatch,
                uint8_t * roots )
{
    return static_cast< SelfPlay * >( h )->Collect( maps, scalars, masks, agents, games, maxBatch, roots );
}

void sp_feed( void * h, float const * logits, float const * values, int batch, int64_t const * updates )
{
    static_cast< SelfPlay * >( h )->Feed( logits, values, batch, updates );
}

//! Read between Collect/Feed calls, when the simulator's worker threads have all joined.
int sp_collect_recurrent(void * h, uint8_t * maps, float * scalars, uint8_t * masks,
                         int * agents, int * games, int maxBatch, uint8_t * roots, float * memory,
                         int * lengths, uint8_t * sequenceMaps, float * sequenceScalars, uint8_t * sequenceMasks,
                         float * sequenceContexts) {
    return static_cast<SelfPlay *>(h)->Collect(maps, scalars, masks, agents, games, maxBatch,
        roots, memory, lengths, sequenceMaps, sequenceScalars, sequenceMasks, sequenceContexts);
}

void sp_feed_recurrent(void * h, float const * logits, float const * values, int batch,
                       int64_t const * updates, float const * memory) {
    static_cast<SelfPlay *>(h)->Feed(logits, values, batch, updates, memory);
}

//! The returned string is owned here and remains valid until this thread's next call.
char const * sp_watch( void const * h, int arena )
{
    SelfPlay const & sp = *static_cast< SelfPlay const * >( h );
    if ( arena < 0 || arena >= int( sp.games.size() ) )
        return 0;
    tron::Game const & g = sp.games[arena];
    World const & w = g.world;
    std::ostringstream out;
    out.imbue( std::locale::classic() );
    out << std::setprecision( std::numeric_limits< float >::max_digits10 );
    out << "{\"arena\":" << arena << ",\"n_arenas\":" << sp.games.size()
        << ",\"decision\":" << g.decision << ",\"time\":" << w.time
        << ",\"active\":" << ( g.active ? "true" : "false" )
        << ",\"finished\":" << ( g.finished ? "true" : "false" )
        << ",\"searching\":" << ( g.searching ? "true" : "false" )
        << ",\"search_sims_done\":" << ( g.searching ? g.simsDone : 0 )
        << ",\"search_sims_budget\":" << sp.cfg.sims
        << ",\"temperature\":" << sp.cfg.temperature << ",\"gumbel\":" << sp.cfg.gumbel
        << ",\"bounds\":[" << w.minX << ',' << w.minY << ',' << w.maxX << ',' << w.maxY << ']'
        << ",\"source_counts\":{\"search\":" << g.sourceCounts[tron::WATCH_SEARCH]
        << ",\"policy_sampled\":" << g.sourceCounts[tron::WATCH_SAMPLED]
        << ",\"greedy\":" << g.sourceCounts[tron::WATCH_GREEDY]
        << ",\"hunter\":" << g.sourceCounts[tron::WATCH_HUNTER] << "},\"players\":[";
    char const * selections[] = { "waiting", "search", "policy_sampled", "greedy", "hunter" };
    for ( int p = 0; p < g.n; ++p )
    {
        if ( p ) out << ',';
        tron::Cycle const & c = w.cycles[p];
        tron::DecisionWatch const & d = g.lastChoice[p];
        out << "{\"id\":" << p << ",\"agent_id\":" << g.agents[p]
            << ",\"learning\":" << ( g.learn[p] ? "true" : "false" )
            << ",\"alive\":" << ( c.alive ? "true" : "false" )
            << ",\"x\":" << c.x << ",\"y\":" << c.y << ",\"dx\":" << c.dx << ",\"dy\":" << c.dy
            << ",\"speed\":" << c.Speed() << ",\"rubber\":" << c.rubber
            << ",\"braking\":" << ( c.braking ? "true" : "false" ) << ",\"kills\":" << c.kills
            << ",\"outcome\":";
        if ( g.endDecision[p] >= 0 ) out << g.outcome[p]; else out << "null";
        out << ",\"action\":";
        if ( d.action >= 0 ) out << d.action; else out << "null";
        out << ",\"decision\":" << d.decision << ",\"selection\":\"" << selections[d.selection]
            << "\",\"policy_update\":";
        if ( d.policyUpdate >= 0 ) out << d.policyUpdate; else out << "null";
        out << ",\"hunter_replanned\":" << ( d.hunterReplanned ? "true" : "false" ) << ",\"mask\":[";
        for ( int a = 0; a < tron::kActions; ++a )
        {
            if ( a ) out << ',';
            out << ( d.mask >> a & 1 ? "true" : "false" );
        }
        out << "],\"policy_probs\":";
        if ( d.selection != tron::WATCH_NONE && d.selection != tron::WATCH_HUNTER ) WatchProbabilities( out, d.policy );
        else out << "null";
        out << ",\"search_probs\":";
        if ( d.selection == tron::WATCH_SEARCH ) WatchProbabilities( out, d.search ); else out << "null";
        out << ",\"value\":";
        if ( d.selection != tron::WATCH_NONE && d.selection != tron::WATCH_HUNTER && std::isfinite( d.value ) )
            out << d.value;
        else out << "null";
        out << ",\"observation\":";
        if ( d.maps.empty() ) out << "null";
        else
        {
            out << "{\"decision\":" << d.decision << ",\"time\":" << d.observedAt << ",\"current\":";
            WatchBytes( out, d.maps );
            out << ",\"previous\":";
            WatchBytes( out, d.previous );
            out << ",\"scalars\":[";
            for ( size_t i = 0; i < d.scalars.size(); ++i )
            {
                if ( i ) out << ',';
                if ( std::isfinite( d.scalars[i] ) ) out << d.scalars[i]; else out << "null";
            }
            out << "],\"context\":[";
            for (int i = 0; i < tron::kContext; ++i) {
                if (i) out << ',';
                out << d.context[i];
            }
            out << "]}";
        }
        out << '}';
    }
    out << "],\"trails\":[";
    for ( int p = 0; p < g.n; ++p )
    {
        if ( p ) out << ',';
        tron::Cycle const & c = w.cycles[p];
        out << "{\"id\":" << p << ",\"alive\":" << ( c.alive ? "true" : "false" )
            << ",\"position\":[" << c.x << ',' << c.y << "],\"heading\":[" << c.dx << ',' << c.dy << ']'
            << ",\"points\":[";
        for ( size_t j = 0; j < c.trail.size(); ++j )
        {
            if ( j ) out << ',';
            out << '[' << c.trail[j].x << ',' << c.trail[j].y << ']';
        }
        if ( !c.trail.empty() ) out << ',';
        out << '[' << c.x << ',' << c.y << "]],\"segments\":";
        WatchSegments( out, w, p );
        out << '}';
    }
    out << "]}";
    static thread_local std::string snapshot;
    snapshot = out.str();
    return snapshot.c_str();
}

//! 1 if round G is over; OUT gets: n, decisions, size, then per player: agent, outcome, died, kills
int sp_game_info( void * h, int g, float * out )
{
    tron::Game const & gm = static_cast< SelfPlay * >( h )->games[g];
    out[0] = float( gm.n );
    out[1] = float( gm.decision );
    out[2] = gm.size;
    for ( int p = 0; p < gm.n; ++p )
    {
        out[3 + 4 * p] = float( gm.agents[p] );
        out[4 + 4 * p] = gm.outcome[p];
        out[5 + 4 * p] = gm.endDecision[p] >= 0 && gm.died[p] ? 1.f : 0.f;
        out[6 + 4 * p] = p < int( gm.world.cycles.size() ) ? float( gm.world.cycles[p].kills ) : 0.f;
    }
    return gm.finished ? 1 : 0;
}

//! counters since the last call: decisions, searches, simulations, samples
void sp_stats( void * h, long long * out )
{
    SelfPlay * s = static_cast< SelfPlay * >( h );
    out[0] = s->statDecisions.exchange( 0 );
    out[1] = s->statSearches.exchange( 0 );
    out[2] = s->statSims.exchange( 0 );
    out[3] = s->statSamples.exchange( 0 );
}
}
