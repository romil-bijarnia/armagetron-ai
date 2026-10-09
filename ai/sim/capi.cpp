// C entry points for Python (ctypes): one World per handle.

#include "tron.h"

#include <cstring>

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
                float temperature, float gumbel, int maxDecisions, int auxHorizon, int threads )
{
    tron::SPConfig & c = static_cast< SelfPlay * >( h )->cfg;
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
}

void sp_set_ring( void * h, int agent, uint8_t * maps, float * scalars, uint8_t * mask, float * policy, uint8_t * hasPolicy,
                  float * value, float * aux, int64_t * prev, int64_t * gen, uint8_t * ready, int64_t * counter, int64_t cap )
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
}

void sp_start( void * h, int g, float size, int n, int const * agents, int const * learn, int rotate, unsigned long long seed )
{
    static_cast< SelfPlay * >( h )->Start( g, size, n, agents, learn, rotate, seed );
}

int sp_collect( void * h, uint8_t * maps, float * scalars, uint8_t * masks, int * agents, int * games, int maxBatch )
{
    return static_cast< SelfPlay * >( h )->Collect( maps, scalars, masks, agents, games, maxBatch );
}

void sp_feed( void * h, float const * logits, float const * values, int batch )
{
    static_cast< SelfPlay * >( h )->Feed( logits, values, batch );
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
