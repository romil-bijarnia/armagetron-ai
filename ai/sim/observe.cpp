// The network's view of the arena: a port of BuildWorld / BuildObservation / BuildTerritory in
// src/tron/gNeural.cpp. Same sampling of walls, same cells, same bits, same numbers.

#include "tron.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tron
{
namespace
{
const int kGrid = World::kGrid;
const int kAgentRow = 40;
const REAL kCloseCell = 1;
const int kRays = 16, kEnemies = 3;
enum { L_WALL = 0, L_OWN, L_ENEMY, L_TEAM, L_ENEMY_HEAD, L_ENEMY_PATH, L_OUTSIDE, L_TEAM_HEAD };
enum { G_WALL = 0, G_OWN, G_ENEMY, G_OWN_HEAD, G_ENEMY_HEAD, G_OUTSIDE, G_TEAM, G_TEAM_HEAD };
enum { T_MINE = 0, T_ENEMY = 1 };

struct Sample
{
    REAL x, y;
    int owner;  // -1: the rim
};

struct WorldView
{
    std::vector< Sample > samples;
    REAL minX, minY, maxX, maxY, cx, cy, extent, step;
    bool Outside( REAL x, REAL y ) const { return x < minX || x > maxX || y < minY || y > maxY; }
};

struct Pose
{
    REAL px, py, hx, hy, rx, ry;
};

inline void SetBit( uint8_t * grid, int row, int col, int bit )
{
    if ( row >= 0 && row < kGrid && col >= 0 && col < kGrid )
        grid[row * kGrid + col] |= uint8_t( 1 << bit );
}

inline void LocalCell( Pose const & f, REAL x, REAL y, int & row, int & col, REAL cell )
{
    REAL dx = x - f.px, dy = y - f.py;
    REAL fw = dx * f.hx + dy * f.hy;
    REAL sd = dx * f.rx + dy * f.ry;
    col = int( std::floor( sd / cell ) ) + kGrid / 2;
    row = kAgentRow - 1 - int( std::floor( fw / cell ) );
}

inline void GlobalCell( WorldView const & w, Pose const & f, REAL x, REAL y, int & row, int & col )
{
    REAL cg = 2 * w.extent / kGrid;
    REAL dx = x - w.cx, dy = y - w.cy;
    REAL fw = dx * f.hx + dy * f.hy;
    REAL sd = dx * f.rx + dy * f.ry;
    col = std::max( 0, std::min( kGrid - 1, int( std::floor( ( sd + w.extent ) / cg ) ) ) );
    row = std::max( 0, std::min( kGrid - 1, int( std::floor( ( w.extent - fw ) / cg ) ) ) );
}

inline REAL Clamp( REAL v, REAL lo, REAL hi )
{
    return v < lo ? lo : ( v > hi ? hi : v );
}

void SampleSegment( WorldView & w, REAL ax, REAL ay, REAL bx, REAL by, int owner,
                    World const * world, REAL d0, REAL d1, REAL time )
{
    REAL ddx = bx - ax, ddy = by - ay;
    REAL len = std::sqrt( ddx * ddx + ddy * ddy );
    int n = int( len / w.step ) + 1;
    for ( int i = 0; i <= n; ++i )
    {
        REAL alpha = REAL( i ) / n;
        if ( owner >= 0 )
        {
            REAL a = std::min( alpha, REAL( .9999 ) );
            if ( !world->WallPointDangerous( owner, d0 + ( d1 - d0 ) * a, time ) )
                continue;
        }
        w.samples.push_back( Sample{ ax + ddx * alpha, ay + ddy * alpha, owner } );
    }
}
}

void World::Observe( int self, uint8_t * local, uint8_t * close, uint8_t * global, uint8_t * territory, float * scalars ) const
{
    int const N = kGrid * kGrid;
    std::memset( local, 0, N );
    std::memset( close, 0, N );
    std::memset( global, 0, N );
    std::memset( territory, 0, N );
    std::memset( scalars, 0, kScalars * sizeof( float ) );

    // ---------------------------------------------------------------- BuildWorld
    WorldView w;
    w.minX = minX;
    w.minY = minY;
    w.maxX = maxX;
    w.maxY = maxY;
    w.cx = .5f * ( minX + maxX );
    w.cy = .5f * ( minY + maxY );
    w.extent = .5f * std::max( maxX - minX, maxY - minY );
    REAL globalCell = 2 * w.extent / kGrid;
    w.step = .5f * std::min( s.localCell, globalCell );
    w.samples.reserve( 4096 );
    // the rim, as the map lists it
    {
        REAL const px[5] = { minX, minX, maxX, maxX, minX }, py[5] = { minY, maxY, maxY, minY, minY };
        for ( int k = 0; k < 4; ++k )
            SampleSegment( w, px[k], py[k], px[k + 1], py[k + 1], -1, this, 0, 0, time );
    }
    for ( int c = 0; c < int( cycles.size() ); ++c )
    {
        Cycle const & cy = cycles[c];
        if ( !cy.alive && time - cy.deathTime > .2f + s.wallsStayUp )
            continue;
        int n = int( cy.trail.size() );
        for ( int seg = 0; seg < n; ++seg )
        {
            REAL x0, y0, d0, x1, y1, d1;
            WallEnds( c, seg, x0, y0, d0, x1, y1, d1, true );
            // IsDangerousAnywhere: the whole wall is behind the trail's end
            if ( s.wallsLength > 0 )
            {
                REAL maxDist = cy.distance - s.wallsLength;
                if ( maxDist > d0 && maxDist > d1 )
                    continue;
            }
            SampleSegment( w, x0, y0, x1, y1, c, this, d0, d1, time );
        }
    }

    // ---------------------------------------------------------------- BuildObservation
    Cycle const & me = cycles[self];
    Pose f;
    f.px = me.x;
    f.py = me.y;
    {
        REAL n = std::sqrt( me.dx * me.dx + me.dy * me.dy );
        f.hx = n > 0 ? me.dx / n : 0;
        f.hy = n > 0 ? me.dy / n : 1;
    }
    f.rx = f.hy;
    f.ry = -f.hx;
    int const team = me.team;
    REAL const c = s.localCell;
    REAL const cg = 2 * w.extent / kGrid;
    REAL const baseSpeed = s.speed > 0 ? s.speed : 20;

    for ( size_t i = 0; i < w.samples.size(); ++i )
    {
        Sample const & sm = w.samples[i];
        int lp = -1, gp = -1;
        if ( sm.owner == self )
            lp = L_OWN, gp = G_OWN;
        else if ( sm.owner >= 0 && cycles[sm.owner].team == team )
            lp = L_TEAM, gp = G_TEAM;
        else if ( sm.owner >= 0 )
            lp = L_ENEMY, gp = G_ENEMY;
        int row, col;
        LocalCell( f, sm.x, sm.y, row, col, c );
        SetBit( local, row, col, L_WALL );
        if ( lp >= 0 )
            SetBit( local, row, col, lp );
        LocalCell( f, sm.x, sm.y, row, col, kCloseCell );
        SetBit( close, row, col, L_WALL );
        if ( lp >= 0 )
            SetBit( close, row, col, lp );
        GlobalCell( w, f, sm.x, sm.y, row, col );
        SetBit( global, row, col, G_WALL );
        if ( gp >= 0 )
            SetBit( global, row, col, gp );
    }

    // outside the arena
    for ( int row = 0; row < kGrid; ++row )
    {
        REAL fw = ( kAgentRow - 1 - row + .5f ) * c;
        REAL fg = w.extent - ( row + .5f ) * cg;
        for ( int col = 0; col < kGrid; ++col )
        {
            REAL sd = ( col - kGrid / 2 + .5f ) * c;
            if ( w.Outside( f.px + f.hx * fw + f.rx * sd, f.py + f.hy * fw + f.ry * sd ) )
                SetBit( local, row, col, L_OUTSIDE );
            REAL fwc = ( kAgentRow - 1 - row + .5f ) * kCloseCell, sdc = ( col - kGrid / 2 + .5f ) * kCloseCell;
            if ( w.Outside( f.px + f.hx * fwc + f.rx * sdc, f.py + f.hy * fwc + f.ry * sdc ) )
                SetBit( close, row, col, L_OUTSIDE );
            REAL sg = ( col + .5f ) * cg - w.extent;
            if ( w.Outside( w.cx + f.hx * fg + f.rx * sg, w.cy + f.hy * fg + f.ry * sg ) )
                SetBit( global, row, col, G_OUTSIDE );
        }
    }

    {
        int row, col;
        GlobalCell( w, f, f.px, f.py, row, col );
        SetBit( global, row, col, G_OWN_HEAD );
    }

    // other cycles, nearest first (the engine walks its player list from the back)
    struct Other
    {
        REAL dist;
        int idx;
        bool enemy;
    };
    std::vector< Other > others;
    int aliveEnemies = 0, aliveMates = 0;
    for ( int i = int( cycles.size() ) - 1; i >= 0; --i )
    {
        if ( i == self || !cycles[i].alive )
            continue;
        REAL ddx = cycles[i].x - f.px, ddy = cycles[i].y - f.py;
        Other o{ std::sqrt( ddx * ddx + ddy * ddy ), i, cycles[i].team != team };
        others.push_back( o );
        if ( o.enemy )
            ++aliveEnemies;
        else
            ++aliveMates;
    }
    std::sort( others.begin(), others.end(), []( Other const & a, Other const & b ) { return a.dist < b.dist; } );

    for ( size_t k = 0; k < others.size(); ++k )
    {
        Cycle const & o = cycles[others[k].idx];
        int row, col;
        LocalCell( f, o.x, o.y, row, col, c );
        SetBit( local, row, col, others[k].enemy ? L_ENEMY_HEAD : L_TEAM_HEAD );
        LocalCell( f, o.x, o.y, row, col, kCloseCell );
        SetBit( close, row, col, others[k].enemy ? L_ENEMY_HEAD : L_TEAM_HEAD );
        GlobalCell( w, f, o.x, o.y, row, col );
        SetBit( global, row, col, others[k].enemy ? G_ENEMY_HEAD : G_TEAM_HEAD );

        if ( others[k].enemy && others[k].dist < 2 * kGrid * c )
        {
            // where the enemy will be within the next second if it keeps going straight
            REAL n = std::sqrt( o.dx * o.dx + o.dy * o.dy );
            REAL ux = n > 0 ? o.dx / n : 0, uy = n > 0 ? o.dy / n : 1;
            Hit h = Cast( others[k].idx, o.x, o.y, ux, uy, 200 );
            REAL reach = std::min( h.t, o.Speed() * REAL( 1 ) );
            for ( REAL t = .5f * kCloseCell; t < reach; t += .5f * kCloseCell )
            {
                REAL qx = o.x + ux * t, qy = o.y + uy * t;
                LocalCell( f, qx, qy, row, col, c );
                SetBit( local, row, col, L_ENEMY_PATH );
                LocalCell( f, qx, qy, row, col, kCloseCell );
                SetBit( close, row, col, L_ENEMY_PATH );
            }
        }
    }

    // ---------------------------------------------------------------- BuildTerritory
    {
        static thread_local std::vector< int > distMine, distEnemy, queue;
        distMine.assign( N, -1 );
        distEnemy.assign( N, -1 );
        queue.clear();
        queue.reserve( N );
        auto blocked = [&]( int idx ) { return ( global[idx] & ( ( 1 << G_WALL ) | ( 1 << G_OUTSIDE ) ) ) != 0; };
        auto bfs = [&]( std::vector< int > & dist, std::vector< int > const & seeds )
        {
            queue.clear();
            for ( size_t i = 0; i < seeds.size(); ++i )
            {
                if ( dist[seeds[i]] >= 0 )
                    continue;
                dist[seeds[i]] = 0;
                queue.push_back( seeds[i] );
            }
            for ( size_t qi = 0; qi < queue.size(); ++qi )
            {
                int idx = queue[qi];
                int r = idx / kGrid, cc = idx % kGrid;
                int const dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
                for ( int k = 0; k < 4; ++k )
                {
                    int rr = r + dr[k], c2 = cc + dc[k];
                    if ( rr < 0 || rr >= kGrid || c2 < 0 || c2 >= kGrid )
                        continue;
                    int j = rr * kGrid + c2;
                    if ( dist[j] >= 0 || blocked( j ) )
                        continue;
                    dist[j] = dist[idx] + 1;
                    queue.push_back( j );
                }
            }
        };
        std::vector< int > mine, enemies;
        {
            int row, col;
            GlobalCell( w, f, f.px, f.py, row, col );
            mine.push_back( row * kGrid + col );
            for ( size_t k = 0; k < others.size(); ++k )
            {
                if ( !others[k].enemy )
                    continue;
                Cycle const & o = cycles[others[k].idx];
                GlobalCell( w, f, o.x, o.y, row, col );
                enemies.push_back( row * kGrid + col );
            }
        }
        bfs( distMine, mine );
        if ( !enemies.empty() )
            bfs( distEnemy, enemies );
        for ( int i = 0; i < N; ++i )
        {
            int dm = distMine[i], de = distEnemy[i];
            if ( blocked( i ) && dm != 0 && de != 0 )
                continue;
            uint8_t v = 0;
            if ( dm >= 0 && ( de < 0 || dm <= de ) )
                v |= 1 << T_MINE;
            if ( de >= 0 && ( dm < 0 || de < dm ) )
                v |= 1 << T_ENEMY;
            if ( dm >= 0 )
                v |= uint8_t( std::min( dm / 4, 63 ) ) << 2;
            territory[i] = v;
        }
    }

    // ---------------------------------------------------------------- scalars
    float * sc = scalars;
    REAL delay = TurnDelay( self );
    sc[0] = me.Speed() / baseSpeed;
    sc[1] = s.rubber > 0 ? me.rubber / s.rubber : 0;
    sc[2] = me.brakingReservoir;
    sc[3] = me.braking ? 1 : 0;
    sc[4] = delay > 0 ? Clamp( ( time - me.LastTurnTime() ) / delay, 0, 5 ) / 5 : 1;
    bool canTurn = me.lastTime >= NextTurn( self );
    sc[5] = canTurn ? 1 : 0;
    sc[6] = canTurn ? 1 : 0;
    sc[7] = 0;  // no queued turns: the network's mask never asks for one
    sc[8] = ( maxX - minX ) / 500;
    sc[9] = ( maxY - minY ) / 500;
    {
        REAL ddx = f.px - w.cx, ddy = f.py - w.cy;
        sc[10] = ( ddx * f.rx + ddy * f.ry ) / w.extent;
        sc[11] = ( ddx * f.hx + ddy * f.hy ) / w.extent;
    }
    sc[12] = Clamp( time, 0, 120 ) / 120;
    sc[13] = aliveEnemies / REAL( 7 );
    sc[14] = aliveMates / REAL( 7 );
    sc[15] = baseSpeed / 20;
    for ( int k = 0; k < kRays; ++k )
    {
        REAL a = REAL( k * 2 * M_PI / kRays );  // clockwise from straight ahead
        REAL ca = REAL( std::cos( a ) ), sa = REAL( std::sin( a ) );
        REAL ddx = f.hx * ca + f.rx * sa, ddy = f.hy * ca + f.ry * sa;
        Hit h = Cast( self, f.px, f.py, ddx, ddy, 1000 );
        REAL dist = h.t;
        sc[16 + 2 * k] = std::min( dist, REAL( 250 ) ) / 250;
        sc[17 + 2 * k] = std::exp( -dist / 8 );
        if ( k % 4 == 0 && h.type >= S_RIM && h.type <= S_SELF )
            sc[48 + ( k / 4 ) * 4 + ( h.type - S_RIM )] = 1;
    }
    int e = 0;
    for ( size_t k = 0; k < others.size() && e < kEnemies; ++k )
    {
        if ( !others[k].enemy )
            continue;
        Cycle const & o = cycles[others[k].idx];
        REAL dpx = o.x - f.px, dpy = o.y - f.py;
        REAL n = std::sqrt( o.dx * o.dx + o.dy * o.dy );
        REAL ox = n > 0 ? o.dx / n : 0, oy = n > 0 ? o.dy / n : 1;
        float * x = sc + 64 + 8 * e;
        x[0] = Clamp( ( dpx * f.hx + dpy * f.hy ) / 100, -3, 3 );
        x[1] = Clamp( ( dpx * f.rx + dpy * f.ry ) / 100, -3, 3 );
        x[2] = Clamp( others[k].dist / 200, 0, 3 );
        x[3] = ox * f.hx + oy * f.hy;
        x[4] = ox * f.rx + oy * f.ry;
        x[5] = o.Speed() / baseSpeed;
        x[6] = o.braking ? 1 : 0;
        x[7] = 1;
        ++e;
    }
}
}
