// The space fighter: see fighter.h.
#include "fighter.h"
#include <algorithm>
#include <vector>

namespace tron
{
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

void FighterPly( World & w, int n, int p, int a, int e, int ea, int macro )
{
    int step[kMaxPlayers];
    for ( int d = 0; d < macro; ++d )
    {
        for ( int q = 0; q < n; ++q )
            step[q] = ACT_STRAIGHT;
        step[p] = d == 0 ? a : Hold( a );
        if ( e >= 0 )
            step[e] = d == 0 ? ea : Hold( ea );
        w.Step( step );
        if ( !w.cycles[p].alive || ( e >= 0 && !w.cycles[e].alive ) )
            return;
    }
}

float FighterEval( World const & w, int p, int e, FighterConfig const & cfg )
{
    if ( !w.cycles[p].alive )
        return -1.f;
    if ( e >= 0 && !w.cycles[e].alive )
        return 1.f;
    static thread_local std::vector< uint8_t > maps;
    static thread_local std::vector< float > sc;
    int const g2 = World::kGrid * World::kGrid;
    maps.resize( 4 * g2 );
    sc.resize( World::kScalars );
    w.Observe( p, &maps[0], &maps[g2], &maps[2 * g2], &maps[3 * g2], &sc[0] );
    float score = 2 * TerritoryShare( &maps[3 * g2] ) - 1;
    // the probe: everyone drives on; a wall met soon costs, an enemy crashing soon pays
    if ( cfg.probe > 0 )
    {
        World probe = w;
        int step[kMaxPlayers];
        for ( int q = 0; q < int( probe.cycles.size() ); ++q )
            step[q] = ACT_STRAIGHT;
        for ( int k = 0; k < cfg.probe; ++k )
        {
            probe.Step( step );
            float soon = 1.f - float( k ) / float( cfg.probe );
            if ( !probe.cycles[p].alive )
                return score - .8f * soon;
            if ( e >= 0 && !probe.cycles[e].alive )
                return score + .3f * soon;
        }
    }
    return score;
}

int FighterMove( World const & w0, int p, FighterConfig const & cfg )
{
    int const n = std::min( int( w0.cycles.size() ), kMaxPlayers );
    if ( p < 0 || p >= n || !w0.cycles[p].alive )
        return ACT_STRAIGHT;
    unsigned mask = w0.ActionMask( p );
    // the nearest living enemy answers; everyone else drives on
    int e = -1;
    REAL bestD = 1e30f;
    for ( int q = 0; q < n; ++q )
    {
        if ( q == p || !w0.cycles[q].alive || w0.cycles[q].team == w0.cycles[p].team )
            continue;
        REAL dx = w0.cycles[q].x - w0.cycles[p].x, dy = w0.cycles[q].y - w0.cycles[p].y;
        REAL d = dx * dx + dy * dy;
        if ( d < bestD )
        {
            bestD = d;
            e = q;
        }
    }
    unsigned emask = e >= 0 ? w0.ActionMask( e ) : 1u;
    int best = ACT_STRAIGHT;
    float bestV = -1e30f;
    for ( int a = 0; a < kActions; ++a )
    {
        if ( !( mask >> a & 1 ) )
            continue;
        float worst = 1e30f;
        for ( int ea = 0; ea < kActions && worst > bestV; ++ea )
        {
            if ( !( emask >> ea & 1 ) )
                continue;
            World w1 = w0;
            FighterPly( w1, n, p, a, e, ea, cfg.ply );
            float v = -1e30f;
            if ( cfg.depth >= 2 && w1.cycles[p].alive && ( e < 0 || w1.cycles[e].alive ) )
            {
                unsigned m2 = w1.ActionMask( p );
                for ( int b = 0; b < kActions; ++b )
                {
                    if ( !( m2 >> b & 1 ) )
                        continue;
                    World w2 = w1;
                    FighterPly( w2, n, p, b, e, ACT_STRAIGHT, cfg.ply );
                    v = std::max( v, FighterEval( w2, p, e, cfg ) );
                }
            }
            if ( v == -1e30f )
                v = FighterEval( w1, p, e, cfg );
            worst = std::min( worst, v );
        }
        if ( worst > bestV )
        {
            bestV = worst;
            best = a;
        }
    }
    return best;
}
}
