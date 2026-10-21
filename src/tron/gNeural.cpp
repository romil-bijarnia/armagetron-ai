/*

*************************************************************************

ArmageTron -- Just another Tron Lightcycle Game in 3D.
Copyright (C) 2000  Manuel Moos (manuel@moosnet.de)

**************************************************************************

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

***************************************************************************

*/

#include "gNeural.h"
#include "gNeuralNet.h"
#include "gNeuralAsync.h"
#include "gFighter.h"
#include "tConsole.h"
#include "tDirectories.h"

#include <chrono>
#include <fstream>
#include <memory>

#include "gAIBase.h"
#include "gAICharacter.h"
#include "gCycle.h"
#include "gWall.h"
#include "gSensor.h"
#include "eAdvWall.h"
#include "ePlayer.h"
#include "eTeam.h"
#include "tConfiguration.h"
#include "tRectangle.h"
#include "nNetwork.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <vector>

extern REAL sg_rubberCycle;

namespace
{
// wire protocol; keep in sync with arma_ai/protocol.py
const uint32_t kMagic = 0x414D5241;
const uint32_t kProtocol = 4;
const uint32_t kMsgHello = 1, kMsgStep = 2, kMsgActions = 3, kMsgWorld = 4;
const int kGrid = 64;
const int kLocalPlanes = 8, kGlobalPlanes = 8, kScalars = 108;
const int kMaps = 4;  // local (2 m), close (1 m), global (arena), territory (arena); one byte per cell each
const REAL kCloseCell = 1;  // the sharp player view: 1 m cells, 64 m across, 40 m ahead
enum { T_MINE = 0, T_ENEMY = 1 };  // territory bits; bits 2..7 hold my BFS distance / 4 cells, capped
const int kAgentRow = 40;  // rows of the local map that lie in front of the cycle
const int kRays = 16;
const int kEnemies = 3;

enum { FLAG_ALIVE = 1, FLAG_NEEDS_ACTION = 2, FLAG_DIED = 4, FLAG_WON = 8 };
enum { ACT_STRAIGHT = 0, ACT_LEFT = 1, ACT_RIGHT = 2, ACT_BRAKE = 3, ACT_LEFT_BRAKE = 4, ACT_RIGHT_BRAKE = 5 };
enum { L_WALL = 0, L_OWN, L_ENEMY, L_TEAM, L_ENEMY_HEAD, L_ENEMY_PATH, L_OUTSIDE, L_TEAM_HEAD };
enum { G_WALL = 0, G_OWN, G_ENEMY, G_OWN_HEAD, G_ENEMY_HEAD, G_OUTSIDE, G_TEAM, G_TEAM_HEAD };

tString sg_socketPath("");
tSettingItemLine sg_socketPathConf( "NEURAL_SOCKET", sg_socketPath );
int sg_engineID = 0;
tSettingItem<int> sg_engineIDConf( "NEURAL_ENGINE_ID", sg_engineID );
int sg_numSlots = 1; // AI players driven by the network: by a trainer over NEURAL_SOCKET, or by the brain in the game
tSettingItem<int> sg_numSlotsConf( "NEURAL_SLOTS", sg_numSlots );
REAL sg_interval = .05;
tSettingItem<REAL> sg_intervalConf( "NEURAL_DECISION_INTERVAL", sg_interval );
REAL sg_localCell = 2;
tSettingItem<REAL> sg_localCellConf( "NEURAL_LOCAL_CELL", sg_localCell );
bool sg_endRound = false;
tSettingItem<bool> sg_endRoundConf( "NEURAL_END_ROUND_WITHOUT_NEURAL", sg_endRound );
REAL sg_lockstepDT = 0;
tSettingItem<REAL> sg_lockstepDTConf( "LOCKSTEP_DT", sg_lockstepDT );
int sg_debug = 0;
tSettingItem<int> sg_debugConf( "NEURAL_DEBUG", sg_debug );
bool sg_controlAfterRound = false;
tSettingItem<bool> sg_controlAfterRoundConf( "NEURAL_CONTROL_AFTER_ROUND", sg_controlAfterRound );
bool sg_spectate = false; // also send every cycle's position each tick, for a match viewer
tSettingItem<bool> sg_spectateConf( "NEURAL_SPECTATE", sg_spectate );
bool sg_teacher = false; // the built-in AI acts; the socket receives its exact pre-action labels
tSettingItem<bool> sg_teacherConf( "NEURAL_TEACHER", sg_teacher );
gAIPlayer const * sg_teacherPlayer = NULL;
int sg_teacherTurns = 0, sg_teacherDirection = 0;
bool sg_teacherValid = true;
REAL sg_roundTimeLimit = 0; // seconds; 0 = none. The tools set it: two cycles circling forever end in a draw
tSettingItem<REAL> sg_roundTimeLimitConf( "NEURAL_ROUND_TIME_LIMIT", sg_roundTimeLimit );
// the brain inside the game, used when no trainer is attached
tString sg_modelPath("brain/policy.bin");
tSettingItemLine sg_modelPathConf( "NEURAL_MODEL", sg_modelPath );
bool sg_sample = false; // draw moves from the policy instead of always taking the best one
tSettingItem<bool> sg_sampleConf( "NEURAL_SAMPLE", sg_sample );
tString sg_name("Brain"); // what the network-driven players are called
tSettingItemLine sg_nameConf( "NEURAL_NAME", sg_name );
tString sg_characterProfile( "" );
tSettingItemLine sg_characterProfileConf( "NEURAL_CHARACTER_PROFILE", sg_characterProfile );
tString sg_builtinProfiles( "" );
tSettingItemLine sg_builtinProfilesConf( "BUILTIN_CHARACTER_PROFILES", sg_builtinProfiles );

// ------------------------------------------------------------------ socket
int sg_fd = -1;

void Fail( char const * what )
{
    std::cerr << "neural bridge: " << what << " (" << strerror( errno ) << "), exiting\n";
    _exit( 1 );
}

void WriteAll( void const * data, size_t len )
{
    char const * p = static_cast< char const * >( data );
    while ( len > 0 )
    {
        ssize_t n = send( sg_fd, p, len, 0 );
        if ( n < 0 && errno == EINTR )
            continue;
        if ( n <= 0 )
            Fail( "send failed" );
        p += n;
        len -= n;
    }
}

void ReadAll( void * data, size_t len )
{
    char * p = static_cast< char * >( data );
    while ( len > 0 )
    {
        ssize_t n = recv( sg_fd, p, len, 0 );
        if ( n < 0 && errno == EINTR )
            continue;
        if ( n == 0 )
        {
            // the brain went away; nothing left to do for us
            _exit( 0 );
        }
        if ( n < 0 )
            Fail( "recv failed" );
        p += n;
        len -= n;
    }
}

class Buffer
{
public:
    std::vector< uint8_t > data;
    void U8( uint8_t v ) { data.push_back( v ); }
    void U32( uint32_t v ) { Raw( &v, 4 ); }
    void F32( float v ) { Raw( &v, 4 ); }
    void Raw( void const * p, size_t n )
    {
        uint8_t const * b = static_cast< uint8_t const * >( p );
        data.insert( data.end(), b, b + n );
    }
};

void SendMessage( uint32_t type, Buffer const & payload )
{
    uint32_t header[3] = { kMagic, type, uint32_t( payload.data.size() ) };
    WriteAll( header, sizeof( header ) );
    if ( !payload.data.empty() )
        WriteAll( &payload.data[0], payload.data.size() );
}

std::string Trimmed( std::string const & s )
{
    size_t b = s.find_first_not_of( " \t\r\n" );
    size_t e = s.find_last_not_of( " \t\r\n" );
    return b == std::string::npos ? std::string() : s.substr( b, e - b + 1 );
}

void Connect()
{
    if ( sg_fd >= 0 )
        return;

    std::string path = Trimmed( sg_socketPath );
    sockaddr_un addr;
    memset( &addr, 0, sizeof( addr ) );
    addr.sun_family = AF_UNIX;
    if ( path.size() >= sizeof( addr.sun_path ) )
        Fail( "NEURAL_SOCKET path too long" );
    strncpy( addr.sun_path, path.c_str(), sizeof( addr.sun_path ) - 1 );

    for ( int tries = 0; ; ++tries )
    {
        sg_fd = socket( AF_UNIX, SOCK_STREAM, 0 );
        if ( sg_fd < 0 )
            Fail( "socket() failed" );
        if ( connect( sg_fd, reinterpret_cast< sockaddr * >( &addr ), sizeof( addr ) ) == 0 )
            break;
        close( sg_fd );
        sg_fd = -1;
        if ( tries > 600 )
            Fail( "could not connect to NEURAL_SOCKET" );
        usleep( 100000 );
    }

    Buffer hello;
    hello.U32( sg_engineID );
    hello.U32( kProtocol );
    hello.U32( sg_numSlots );
    hello.U32( kGrid );
    hello.U32( kLocalPlanes );
    hello.U32( kGlobalPlanes );
    hello.U32( kScalars );
    hello.U32( kMaps );
    hello.F32( sg_localCell );
    hello.F32( sg_interval );
    SendMessage( kMsgHello, hello );
}

// ------------------------------------------------------------------ world snapshot
struct Sample
{
    REAL x, y;
    gCycle const * owner; // NULL for rim walls
};

struct World
{
    std::vector< Sample > samples;
    REAL minX, minY, maxX, maxY; // arena bounds
    REAL cx, cy, extent;         // centre and half size of the square global map
    REAL step;                   // sample spacing along walls

    bool Outside( REAL x, REAL y ) const
    {
        return x < minX || x > maxX || y < minY || y > maxY;
    }
};

void SampleSegment( World & w, eCoord const & a, eCoord const & b, gCycle const * owner,
                    gNetPlayerWall const * wall, REAL time )
{
    eCoord d = b - a;
    REAL len = d.Norm();
    int n = int( len / w.step ) + 1;
    for ( int i = 0; i <= n; ++i )
    {
        REAL alpha = REAL( i ) / n;
        if ( wall && !wall->IsDangerous( std::min( alpha, REAL( .9999 ) ), time ) )
            continue;
        eCoord p = a + d * alpha;
        Sample s = { p.x, p.y, owner };
        w.samples.push_back( s );
    }
}

void SampleWalls( World & w, tList< gNetPlayerWall > & list, REAL time )
{
    for ( int i = list.Len() - 1; i >= 0; --i )
    {
        gNetPlayerWall * wall = list( i );
        if ( !wall || !wall->Cycle() || !wall->IsDangerousAnywhere( time ) )
            continue;
        SampleSegment( w, wall->EndPoint( 0 ), wall->EndPoint( 1 ), wall->Cycle(), wall, time );
    }
}

void BuildWorld( World & w, REAL time )
{
    tRectangle const & bounds = eWallRim::GetBounds();
    w.minX = bounds.GetLow().x;
    w.minY = bounds.GetLow().y;
    w.maxX = bounds.GetHigh().x;
    w.maxY = bounds.GetHigh().y;
    w.cx = .5 * ( w.minX + w.maxX );
    w.cy = .5 * ( w.minY + w.maxY );
    w.extent = .5 * std::max( w.maxX - w.minX, w.maxY - w.minY );
    REAL globalCell = 2 * w.extent / kGrid;
    w.step = .5 * std::min( sg_localCell, globalCell );
    w.samples.clear();

    for ( int i = se_rimWalls.Len() - 1; i >= 0; --i )
    {
        eWallRim * rim = se_rimWalls( i );
        if ( rim && rim->Edge() )
            SampleSegment( w, rim->EndPoint( 0 ), rim->EndPoint( 1 ), NULL, NULL, time );
    }
    SampleWalls( w, sg_netPlayerWalls, time );
    SampleWalls( w, sg_netPlayerWallsGridded, time );
}

// ------------------------------------------------------------------ observation
struct Frame
{
    eCoord pos, h, r; // position, heading, right-hand side
};

inline void SetBit( uint8_t * grid, int row, int col, int bit )
{
    if ( row >= 0 && row < kGrid && col >= 0 && col < kGrid )
        grid[ row * kGrid + col ] |= uint8_t( 1 << bit );
}

inline void LocalCell( Frame const & f, REAL x, REAL y, int & row, int & col, REAL cell = sg_localCell )
{
    REAL dx = x - f.pos.x, dy = y - f.pos.y;
    REAL fw = dx * f.h.x + dy * f.h.y;
    REAL sd = dx * f.r.x + dy * f.r.y;
    col = int( floor( sd / cell ) ) + kGrid / 2;
    row = kAgentRow - 1 - int( floor( fw / cell ) );
}

inline void GlobalCell( World const & w, Frame const & f, REAL x, REAL y, int & row, int & col )
{
    REAL cg = 2 * w.extent / kGrid;
    REAL dx = x - w.cx, dy = y - w.cy;
    REAL fw = dx * f.h.x + dy * f.h.y;
    REAL sd = dx * f.r.x + dy * f.r.y;
    col = std::max( 0, std::min( kGrid - 1, int( floor( ( sd + w.extent ) / cg ) ) ) );
    row = std::max( 0, std::min( kGrid - 1, int( floor( ( w.extent - fw ) / cg ) ) ) );
}

inline REAL Clamp( REAL v, REAL lo, REAL hi )
{
    return v < lo ? lo : ( v > hi ? hi : v );
}

eCoord Unit( eCoord v )
{
    REAL n = v.Norm();
    return n > 0 ? v * ( 1 / n ) : eCoord( 0, 1 );
}

REAL Ray( gCycle * self, eCoord const & pos, eCoord const & dir, REAL range, int & type )
{
    gSensor sensor( self, pos, dir );
    sensor.detect( range );
    type = sensor.type;
    return sensor.hit;
}

struct Other
{
    REAL dist;
    gCycle * cycle;
    bool enemy;
};

// ------------------------------------------------------------------ territory
// A breadth-first race over the arena grid: from my head and from every enemy head, who reaches
// each free cell first. Walls block; it is the Voronoi partition every good Tron player holds in
// their head when they glance at the minimap.
void BuildTerritory( World const & w, Frame const & f, uint8_t const * global, gCycle * self,
                     std::vector< Other > const & others, uint8_t * territory )
{
    int const N = kGrid * kGrid;
    memset( territory, 0, N );
    static std::vector< int > distMine, distEnemy, queue;
    distMine.assign( N, -1 );
    distEnemy.assign( N, -1 );
    queue.clear();
    queue.reserve( N );

    // a head's own cell holds the start of its trail, so a seed is never "blocked"; the frontier
    // of its trail's neighbours is what matters
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
            int r = idx / kGrid, c = idx % kGrid;
            int const dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
            for ( int k = 0; k < 4; ++k )
            {
                int rr = r + dr[k], cc = c + dc[k];
                if ( rr < 0 || rr >= kGrid || cc < 0 || cc >= kGrid )
                    continue;
                int j = rr * kGrid + cc;
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
        GlobalCell( w, f, f.pos.x, f.pos.y, row, col );
        mine.push_back( row * kGrid + col );
        for ( size_t i = 0; i < others.size(); ++i )
        {
            if ( !others[i].enemy )
                continue;
            eCoord p = others[i].cycle->Position();
            GlobalCell( w, f, p.x, p.y, row, col );
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
            continue;  // walls own nothing; the heads' own cells keep their owner
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

void BuildObservation( World const & w, gCycle * self, REAL time, int aliveEnemies, int aliveMates,
                       uint8_t * local, uint8_t * close, uint8_t * global, uint8_t * territory, float * scalars )
{
    memset( local, 0, kGrid * kGrid );
    memset( close, 0, kGrid * kGrid );
    memset( global, 0, kGrid * kGrid );
    memset( scalars, 0, kScalars * sizeof( float ) );

    Frame f;
    f.pos = self->Position();
    f.h = Unit( self->Direction() );
    f.r = eCoord( f.h.y, -f.h.x );
    eTeam const * team = self->Team();
    REAL c = sg_localCell;
    REAL cg = 2 * w.extent / kGrid;
    REAL baseSpeed = gCycleMovement::BaseSpeed();
    if ( baseSpeed <= 0 )
        baseSpeed = 20;

    // walls
    for ( size_t i = 0; i < w.samples.size(); ++i )
    {
        Sample const & s = w.samples[i];
        int lp = -1, gp = -1;
        if ( s.owner == self )
            lp = L_OWN, gp = G_OWN;
        else if ( s.owner && team && s.owner->Team() == team )
            lp = L_TEAM, gp = G_TEAM;
        else if ( s.owner )
            lp = L_ENEMY, gp = G_ENEMY;

        int row, col;
        LocalCell( f, s.x, s.y, row, col );
        SetBit( local, row, col, L_WALL );
        if ( lp >= 0 )
            SetBit( local, row, col, lp );
        LocalCell( f, s.x, s.y, row, col, kCloseCell );
        SetBit( close, row, col, L_WALL );
        if ( lp >= 0 )
            SetBit( close, row, col, lp );
        GlobalCell( w, f, s.x, s.y, row, col );
        SetBit( global, row, col, G_WALL );
        if ( gp >= 0 )
            SetBit( global, row, col, gp );
    }

    // outside the arena
    for ( int row = 0; row < kGrid; ++row )
    {
        REAL fw = ( kAgentRow - 1 - row + .5 ) * c;
        REAL fg = w.extent - ( row + .5 ) * cg;
        for ( int col = 0; col < kGrid; ++col )
        {
            REAL sd = ( col - kGrid / 2 + .5 ) * c;
            eCoord p = f.pos + f.h * fw + f.r * sd;
            if ( w.Outside( p.x, p.y ) )
                SetBit( local, row, col, L_OUTSIDE );
            REAL fwc = ( kAgentRow - 1 - row + .5 ) * kCloseCell, sdc = ( col - kGrid / 2 + .5 ) * kCloseCell;
            eCoord pc = f.pos + f.h * fwc + f.r * sdc;
            if ( w.Outside( pc.x, pc.y ) )
                SetBit( close, row, col, L_OUTSIDE );
            REAL sg = ( col + .5 ) * cg - w.extent;
            eCoord q = eCoord( w.cx, w.cy ) + f.h * fg + f.r * sg;
            if ( w.Outside( q.x, q.y ) )
                SetBit( global, row, col, G_OUTSIDE );
        }
    }

    {
        int row, col;
        GlobalCell( w, f, f.pos.x, f.pos.y, row, col );
        SetBit( global, row, col, G_OWN_HEAD );
    }

    // other cycles
    std::vector< Other > others;
    for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
    {
        gCycle * o = dynamic_cast< gCycle * >( se_PlayerNetIDs( i )->Object() );
        if ( !o || o == self || !o->Alive() )
            continue;
        Other x;
        x.cycle = o;
        x.dist = ( o->Position() - f.pos ).Norm();
        x.enemy = !team || o->Team() != team;
        others.push_back( x );
    }
    std::sort( others.begin(), others.end(), []( Other const & a, Other const & b ) { return a.dist < b.dist; } );

    for ( size_t i = 0; i < others.size(); ++i )
    {
        gCycle * o = others[i].cycle;
        eCoord p = o->Position();
        int row, col;
        LocalCell( f, p.x, p.y, row, col );
        SetBit( local, row, col, others[i].enemy ? L_ENEMY_HEAD : L_TEAM_HEAD );
        LocalCell( f, p.x, p.y, row, col, kCloseCell );
        SetBit( close, row, col, others[i].enemy ? L_ENEMY_HEAD : L_TEAM_HEAD );
        GlobalCell( w, f, p.x, p.y, row, col );
        SetBit( global, row, col, others[i].enemy ? G_ENEMY_HEAD : G_TEAM_HEAD );

        if ( others[i].enemy && others[i].dist < 2 * kGrid * c )
        {
            // where the enemy will be within the next second if it keeps going straight
            eCoord d = Unit( o->Direction() );
            int type;
            REAL reach = std::min( Ray( o, p, d, 200, type ), o->Speed() * REAL( 1 ) );
            for ( REAL t = .5 * kCloseCell; t < reach; t += .5 * kCloseCell )
            {
                eCoord q = p + d * t;
                LocalCell( f, q.x, q.y, row, col );
                SetBit( local, row, col, L_ENEMY_PATH );
                LocalCell( f, q.x, q.y, row, col, kCloseCell );
                SetBit( close, row, col, L_ENEMY_PATH );
            }
        }
    }

    BuildTerritory( w, f, global, self, others, territory );

    // scalars
    float * s = scalars;
    REAL delay = self->GetTurnDelay();
    s[0] = self->Speed() / baseSpeed;
    s[1] = sg_rubberCycle > 0 ? self->GetRubber() / sg_rubberCycle : 0;
    s[2] = self->GetBrakingReservoir();
    s[3] = self->GetBraking() ? 1 : 0;
    s[4] = delay > 0 ? Clamp( ( time - self->GetLastTurnTime() ) / delay, 0, 5 ) / 5 : 1;
    s[5] = self->CanMakeTurn( -1 ) ? 1 : 0;
    s[6] = self->CanMakeTurn( 1 ) ? 1 : 0;
    s[7] = Clamp( self->PendingTurnCount() / REAL( 3 ), 0, 1 );
    s[8] = ( w.maxX - w.minX ) / 500;
    s[9] = ( w.maxY - w.minY ) / 500;
    {
        REAL dx = f.pos.x - w.cx, dy = f.pos.y - w.cy;
        s[10] = ( dx * f.r.x + dy * f.r.y ) / w.extent;
        s[11] = ( dx * f.h.x + dy * f.h.y ) / w.extent;
    }
    s[12] = Clamp( time, 0, 120 ) / 120;
    s[13] = aliveEnemies / REAL( 7 );
    s[14] = aliveMates / REAL( 7 );
    s[15] = baseSpeed / 20;

    for ( int k = 0; k < kRays; ++k )
    {
        REAL a = k * 2 * M_PI / kRays; // clockwise from straight ahead
        eCoord d = f.h * REAL( cos( a ) ) + f.r * REAL( sin( a ) );
        int type;
        REAL dist = Ray( self, f.pos, d, 1000, type );
        s[16 + 2 * k] = std::min( dist, REAL( 250 ) ) / 250;
        s[17 + 2 * k] = exp( -dist / 8 );
        if ( k % 4 == 0 && type >= gSENSOR_RIM && type <= gSENSOR_SELF )
            s[48 + ( k / 4 ) * 4 + ( type - gSENSOR_RIM )] = 1;
    }

    int e = 0;
    for ( size_t i = 0; i < others.size() && e < kEnemies; ++i )
    {
        if ( !others[i].enemy )
            continue;
        gCycle * o = others[i].cycle;
        eCoord dp = o->Position() - f.pos;
        eCoord od = Unit( o->Direction() );
        float * x = s + 64 + 8 * e;
        x[0] = Clamp( ( dp.x * f.h.x + dp.y * f.h.y ) / 100, -3, 3 );
        x[1] = Clamp( ( dp.x * f.r.x + dp.y * f.r.y ) / 100, -3, 3 );
        x[2] = Clamp( others[i].dist / 200, 0, 3 );
        x[3] = od.x * f.h.x + od.y * f.h.y;
        x[4] = od.x * f.r.x + od.y * f.r.y;
        x[5] = o->Speed() / baseSpeed;
        x[6] = o->GetBraking() ? 1 : 0;
        x[7] = 1;
        // Preserve the original 96 scalars and append four resources/timers per enemy in
        // exactly the same nearest-enemy order. Slots 88..95 remain reserved and zero.
        float * extra = s + 96 + 4 * e;
        extra[0] = o->GetBrakingReservoir();
        extra[1] = sg_rubberCycle > 0 ? o->GetRubber() / sg_rubberCycle : 0;
        REAL const turnDelay = std::max( o->GetTurnDelay(), REAL( 1e-6 ) );
        extra[2] = Clamp( ( o->GetNextTurn( -1 ) - o->LastTime() ) / turnDelay, 0, 1 );
        extra[3] = Clamp( ( o->GetNextTurn( 1 ) - o->LastTime() ) / turnDelay, 0, 1 );
        ++e;
    }
}

uint8_t ActionMask( gCycle * c )
{
    uint8_t mask = ( 1 << ACT_STRAIGHT ) | ( 1 << ACT_BRAKE );
    bool idle = c->PendingTurnCount() == 0;
    if ( idle && c->CanMakeTurn( -1 ) )
        mask |= ( 1 << ACT_LEFT ) | ( 1 << ACT_LEFT_BRAKE );
    if ( idle && c->CanMakeTurn( 1 ) )
        mask |= ( 1 << ACT_RIGHT ) | ( 1 << ACT_RIGHT_BRAKE );
    return mask;
}

int Apply( gCycle * c, int action )
{
    unsigned short const beforeTurns = c->GetTurns();
    bool brake = action == ACT_BRAKE || action == ACT_LEFT_BRAKE || action == ACT_RIGHT_BRAKE;
    if ( ( c->GetBraking() != 0 ) != brake )
        c->Act( &gCycle::s_brake, brake ? 1 : -1 );
    if ( action == ACT_LEFT || action == ACT_LEFT_BRAKE )
        c->Act( &gCycle::se_turnLeft, 1 );
    else if ( action == ACT_RIGHT || action == ACT_RIGHT_BRAKE )
        c->Act( &gCycle::se_turnRight, 1 );
    // At the initial object-creation decision the game has not started accepting
    // controls. Report the executed result, never merely the requested turn.
    if ( c->LastTime() <= 0 ) return -1;
    bool const executedBrake = c->GetBraking() != 0;
    if ( c->GetTurns() != beforeTurns )
        return action == ACT_LEFT || action == ACT_LEFT_BRAKE ?
               ( executedBrake ? ACT_LEFT_BRAKE : ACT_LEFT ) :
               ( executedBrake ? ACT_RIGHT_BRAKE : ACT_RIGHT );
    return executedBrake ? ACT_BRAKE : ACT_STRAIGHT;
}

// ------------------------------------------------------------------ the brain in the game
gNeuralNet::Net sg_net;
std::string sg_netPath; // the NEURAL_MODEL the last load attempt was for
using LocalWorker = gNeuralAsync::Worker< gNeuralNet::Net::Memory >;
std::unique_ptr< LocalWorker > sg_localWorker;
uint64_t sg_weightGeneration = 0, sg_lifeGeneration = 0;
uint64_t sg_localApplied = 0, sg_localDiscarded = 0, sg_lastWorkerReport = 0;
double sg_localApplyLatencyMs = 0;
bool sg_shuttingDown = false;

void StopLocalWorker()
{
    // Joins only for explicit configuration/model transitions or application shutdown.
    // The ordinary frame loop never waits for inference.
    sg_localWorker.reset();
    sg_localApplied = sg_localDiscarded = sg_lastWorkerReport = 0;
    sg_localApplyLatencyMs = 0;
}

int RunLocalInference( gNeuralAsync::Observation const & observation, gNeuralNet::Net::Memory & memory,
                       uint32_t & rng )
{
    // Worker thread: sg_net's weights stay immutable until this worker is joined. Everything
    // else comes from the copied job; no game objects, SDL, OpenGL or mutable settings here.
    size_t cells = size_t( sg_net.Grid() ) * sg_net.Grid();
    if ( observation.maps.size() != 4 * cells || observation.scalars.size() != size_t( sg_net.NumScalars() ) )
        throw std::runtime_error( "local inference observation dimensions do not match the loaded policy" );
    uint8_t const * maps[4];
    for ( size_t m = 0; m < 4; ++m ) maps[m] = observation.maps.data() + m * cells;
    float probs[6], value;
    return sg_net.Decide( maps, observation.scalars.data(), observation.mask, observation.sample,
                         probs, &value, rng, memory, observation.context.data() );
}

bool SocketMode()
{
    return !Trimmed( sg_socketPath ).empty();
}

//! true when no trainer is attached and the policy file could be loaded (loaded on first use)
bool LocalBrain()
{
    if ( SocketMode() )
        return false;
    std::string path = Trimmed( sg_modelPath );
    if ( path.empty() )
    {
        StopLocalWorker();
        return false;
    }
    if ( path != sg_netPath )
    {
        StopLocalWorker(); // never mutate model weights while an inference can still read them
        ++sg_weightGeneration;
        sg_netPath = path;
        sg_net = gNeuralNet::Net();
        std::ifstream f;
        std::string error;
        if ( !tDirectories::Data().Open( f, path.c_str() ) )
            con << "Neural brain: " << path << " not found in the data folders, the AIs stay built in.\n";
        else if ( !sg_net.Load( f, error ) )
            con << "Neural brain: could not read " << path << ": " << error << "\n";
        else
            con << "Neural brain: " << path << " loaded (training update " << sg_net.Update() << ").\n";
    }
    if ( sg_net.Loaded() && ( !sg_localWorker || sg_localWorker->Capacity() != size_t( sg_numSlots ) ) )
    {
        StopLocalWorker();
        sg_localWorker.reset( new LocalWorker( size_t( sg_numSlots ), RunLocalInference ) );
    }
    return sg_net.Loaded();
}

void Rename( ePlayerNetID * ai, size_t slot )
{
    tString want( sg_name );
    if ( slot > 0 )
        want << " " << int( slot + 1 );
    if ( want.Len() > 1 && ai->GetName() != want )
        ai->SetName( want );
    // all AIs of a local game share one team, named after its oldest member; make sure it is
    // called after the brain, not after whichever built-in bot happened to join first
    if ( slot == 0 )
    {
        ai->SetTeamname( want );
        if ( ai->CurrentTeam() )
            ai->CurrentTeam()->UpdateAppearance();
    }
}

// ------------------------------------------------------------------ bookkeeping
struct Slot
{
    ePlayerNetID * player;
    bool wasAlive;
    // Main-thread observers invalidate even if a destroyed object's address gets reused.
    nObserverPtr< ePlayerNetID > localPlayer;
    nObserverPtr< gCycle > localCycle;
    bool localAlive = false;
    uint64_t life = 0, lastApplied = 0;
    gNeuralAsync::ControlContext context;
    gNeuralAsync::Ticket latest;
    Slot(): player( NULL ), wasAlive( false ) {}
};

std::vector< Slot > sg_slots;
std::map< ePlayerNetID const *, bool > sg_wasAlive; // every player, for kill accounting
uint32_t sg_roundID = 0;
uint32_t sg_tick = 0;
bool sg_roundOver = false;
bool sg_roundStarted = false;
int sg_roundTotal = 0;
REAL sg_nextDecision = 0;

bool Listed( ePlayerNetID const * p )
{
    for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
        if ( se_PlayerNetIDs( i ) == p )
            return true;
    return false;
}

void AssignSlots()
{
    if ( int( sg_slots.size() ) != sg_numSlots )
        sg_slots.resize( std::max( 0, sg_numSlots ) );
    // a slot whose player left the game, or lost its cycle for good (the AI of a finished match
    // lingers until the next one is set up), is free again
    for ( size_t k = 0; k < sg_slots.size(); ++k )
        if ( sg_slots[k].player && ( !Listed( sg_slots[k].player ) || !sg_slots[k].player->CurrentTeam() ) )
            sg_slots[k] = Slot();
    for ( int i = 0; i < se_PlayerNetIDs.Len(); ++i )
    {
        gAIPlayer * ai = dynamic_cast< gAIPlayer * >( se_PlayerNetIDs( i ) );
        if ( !ai )
            continue;
        bool taken = false;
        for ( size_t k = 0; k < sg_slots.size(); ++k )
            taken = taken || sg_slots[k].player == ai;
        if ( taken )
            continue;
        for ( size_t k = 0; k < sg_slots.size(); ++k )
        {
            if ( !sg_slots[k].player )
            {
                sg_slots[k].player = ai;
                sg_slots[k].wasAlive = false;
                if ( !SocketMode() )
                    Rename( ai, k );
                break;
            }
        }
    }
}

gCycle * SlotCycle( size_t k )
{
    ePlayerNetID * p = sg_slots[k].player;
    return p ? dynamic_cast< gCycle * >( p->Object() ) : NULL;
}

void RefreshLocalLives()
{
    for ( size_t k = 0; k < sg_slots.size(); ++k )
    {
        Slot & slot = sg_slots[k];
        gCycle * cycle = SlotCycle( k );
        bool alive = cycle && cycle->Alive();
        if ( slot.localPlayer.get() != slot.player || slot.localCycle.get() != cycle || slot.localAlive != alive )
        {
            slot.life = ++sg_lifeGeneration;
            slot.lastApplied = 0;
            slot.context.Reset();
            slot.latest = gNeuralAsync::Ticket();
            slot.localPlayer = slot.player;
            slot.localCycle = cycle;
            slot.localAlive = alive;
            if ( sg_localWorker ) sg_localWorker->Invalidate( k );
        }
    }
}

void ApplyLocalResults( REAL time )
{
    if ( !sg_localWorker ) return;
    for ( gNeuralAsync::Result const & result : sg_localWorker->Poll() )
    {
        size_t k = result.ticket.slot;
        if ( k >= sg_slots.size() ) { ++sg_localDiscarded; continue; }
        Slot & slot = sg_slots[k];
        gCycle * cycle = SlotCycle( k );
        bool alive = cycle && cycle->Alive() && slot.localCycle.get() == cycle;
        unsigned legal = alive ? ActionMask( cycle ) : 0;
        if ( result.ticket.round != sg_roundID || result.ticket.life != slot.life ||
             result.ticket.weights != sg_weightGeneration ||
             !gNeuralAsync::Fresh( result, time, sg_interval ) ||
             !gNeuralAsync::CanApply( result, slot.latest, slot.lastApplied, alive, legal ) )
        {
            ++sg_localDiscarded;
            if ( !result.error.empty() ) std::cerr << "neural worker: " << result.error << '\n';
            continue;
        }
        int const executed = Apply( cycle, result.action );
        slot.context.Applied( executed );
        slot.lastApplied = result.ticket.decision;
        if ( executed < 0 ) { ++sg_localDiscarded; continue; }
        ++sg_localApplied;
        sg_localApplyLatencyMs += std::chrono::duration< double, std::milli >(
            std::chrono::steady_clock::now() - result.submitted ).count();
    }
    gNeuralAsync::Statistics stats = sg_localWorker->Stats();
    if ( sg_debug > 0 && stats.processed >= sg_lastWorkerReport + 400 )
    {
        std::cerr << "neural worker: " << stats.computeMs / stats.processed << " ms inference, "
                  << stats.latencyMs / stats.processed << " ms queued+inference; applied=" << sg_localApplied
                  << " apply_latency_ms=" << ( sg_localApplied ? sg_localApplyLatencyMs / sg_localApplied : 0 )
                  << " replaced=" << stats.replaced << " stale=" << stats.stale + sg_localDiscarded
                  << " life_resets=" << stats.lifeResets << " gap_resets=" << stats.gapResets
                  << " skipped_decisions=" << stats.skippedDecisions << '\n';
        sg_lastWorkerReport = stats.processed;
    }
}

class TeacherScope
{
public:
    explicit TeacherScope( gAIPlayer const * player )
    {
        sg_teacherPlayer = player;
        sg_teacherTurns = sg_teacherDirection = 0;
        sg_teacherValid = true;
    }
    ~TeacherScope() { sg_teacherPlayer = NULL; }
};

//! Called only after every slot's observation and mask have been serialized. The label comes
//! from the successful native turn call, never from a later position or the requested action.
uint8_t TeacherAction( size_t k, REAL time, uint8_t mask )
{
    gAIPlayer * teacher = dynamic_cast< gAIPlayer * >( sg_slots[k].player );
    gCycle * c = SlotCycle( k );
    if ( !teacher || !teacher->Character() || !c || !c->Alive() || c->PendingTurnCount() ||
         gSimpleAIFactory::Get() )
        Fail( "teacher requires a live built-in AI with no pending turns or custom AI factory" );
    unsigned short const beforeTurns = c->GetTurns();
    {
        // Normal world timesteps keep this AI suppressed. Its own reaction-time schedule and
        // tactical controller run here once at the policy boundary; emergency help stays off
        // between boundaries, so the recorded controls alone reproduce the trajectory.
        TeacherScope scope( teacher );
        teacher->Timestep( time );
    }
    unsigned short expectedTurns = beforeTurns;
    if ( sg_teacherTurns == 1 )
        expectedTurns = static_cast< unsigned short >( ( beforeTurns ? beforeTurns : 1 ) + 1 );
    if ( !sg_teacherValid || sg_teacherTurns > 1 || teacher->Object() != c || !c->Alive() ||
         c->PendingTurnCount() || c->GetTurns() != expectedTurns )
    {
        std::cerr << "teacher rejected slot=" << k << " time=" << time << " valid=" << sg_teacherValid
                  << " native_turns=" << sg_teacherTurns << " direction=" << sg_teacherDirection
                  << " before_turns=" << beforeTurns << " after_turns=" << c->GetTurns()
                  << " expected_turns=" << expectedTurns << " pending=" << c->PendingTurnCount()
                  << " alive=" << c->Alive() << "\n";
        Fail( "teacher action is ambiguous or could not execute at the policy boundary" );
    }
    bool const brake = c->GetBraking() != 0;
    int action = brake ? ACT_BRAKE : ACT_STRAIGHT;
    if ( sg_teacherDirection < 0 )
        action = brake ? ACT_LEFT_BRAKE : ACT_LEFT;
    else if ( sg_teacherDirection > 0 )
        action = brake ? ACT_RIGHT_BRAKE : ACT_RIGHT;
    if ( !( mask & ( 1 << action ) ) )
        Fail( "teacher executed an action outside the pre-action legal mask" );
    return uint8_t( action );
}
} // namespace

bool gNeural::Active()
{
    if ( sg_shuttingDown || sg_numSlots <= 0 )
    {
        if ( sg_localWorker ) StopLocalWorker();
        return false;
    }
    if ( SocketMode() )
    {
        if ( sg_localWorker ) StopLocalWorker();
        return true;
    }
    return LocalBrain();
}

void gNeural::Shutdown()
{
    sg_shuttingDown = true;
    StopLocalWorker();
    gFighter::Shutdown();
}

int & gNeural::Slots()
{
    return sg_numSlots;
}

unsigned gNeural::LegalActions( gCycle * c )
{
    return ActionMask( c );
}

int gNeural::ApplyAction( gCycle * c, int action )
{
    return Apply( c, action );
}

REAL gNeural::LockstepDT()
{
    return sg_lockstepDT;
}

bool gNeural::Controls( gAIPlayer const * player )
{
    if ( player == sg_teacherPlayer )
        return false;
    if ( !Active() )
        return false;
    if ( int( sg_slots.size() ) != sg_numSlots )
        AssignSlots();
    for ( size_t k = 0; k < sg_slots.size(); ++k )
        if ( sg_slots[k].player == player )
            return true;
    return false;
}

void gNeural::TeacherTurn( gAIPlayer const * player, int direction, bool executed )
{
    if ( player != sg_teacherPlayer )
        return;
    if ( !executed || direction == 0 )
        sg_teacherValid = false;
    else
    {
        ++sg_teacherTurns;
        // gCycle::DoTurn clamps the controller's signed direction to one left/right turn.
        sg_teacherDirection = direction < 0 ? -1 : 1;
    }
}

void gNeural::ConfigureCharacters()
{
    if (sn_GetNetState() == nCLIENT) return;
    std::string const controlledName = Trimmed(sg_characterProfile);
    std::istringstream roster(Trimmed(sg_builtinProfiles));
    std::vector<std::string> names;
    for (std::string name; roster >> name;) names.push_back(name);
    if (controlledName.empty() && names.empty()) return;
    if (controlledName.find_first_of(" \t\r\n") != std::string::npos)
        Fail("NEURAL_CHARACTER_PROFILE requires one exact profile name");
    if ((!controlledName.empty() || sg_numSlots > 0) && !Active())
        Fail("named AI rosters require active neural control for their controlled slots");
    AssignSlots();
    std::set<gAIPlayer *> controlled;
    for (size_t k = 0; k < sg_slots.size(); ++k)
    {
        gAIPlayer * player = dynamic_cast<gAIPlayer *>(sg_slots[k].player);
        if (!player || !player->CurrentTeam())
            Fail("named AI roster is missing a configured neural player slot");
        controlled.insert(player);
        if (!controlledName.empty())
        {
            if (!player->SetCharacterProfile(tString(controlledName.c_str())))
                Fail("NEURAL_CHARACTER_PROFILE is missing or duplicated in AI_CHARACTER_FILE");
            std::cerr << "AI roster: role=controlled slot=" << k << " profile=" << controlledName << '\n';
        }
    }
    if (!controlledName.empty() && controlled.empty())
        Fail("NEURAL_CHARACTER_PROFILE requires at least one controlled player slot");
    std::vector<gAIPlayer *> builtin;
    for (int i = 0; i < se_PlayerNetIDs.Len(); ++i)
    {
        gAIPlayer * player = dynamic_cast<gAIPlayer *>(se_PlayerNetIDs(i));
        if (player && player->CurrentTeam() && !controlled.count(player)) builtin.push_back(player);
    }
    if (!names.empty() && names.size() != builtin.size())
        Fail("BUILTIN_CHARACTER_PROFILES must name every ordinary AI player exactly once");
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (!builtin[i]->SetCharacterProfile(tString(names[i].c_str())))
            Fail("a BUILTIN_CHARACTER_PROFILES name is missing or duplicated in AI_CHARACTER_FILE");
        std::cerr << "AI roster: role=builtin slot=" << i << " profile=" << names[i] << '\n';
    }
}

void gNeural::NewRound()
{
    if ( !Active() )
        return;
    ++sg_roundID;
    sg_tick = 0;
    sg_roundOver = false;
    sg_roundStarted = false;
    sg_roundTotal = 0;
    sg_nextDecision = 0;
    sg_wasAlive.clear();
    if ( sg_localWorker ) sg_localWorker->InvalidateAll();
    AssignSlots();
    for ( size_t k = 0; k < sg_slots.size(); ++k )
    {
        sg_slots[k].wasAlive = false;
        sg_slots[k].localAlive = false;
        sg_slots[k].localCycle = NULL;
        sg_slots[k].life = ++sg_lifeGeneration;
        sg_slots[k].lastApplied = 0;
        sg_slots[k].context.Reset();
        sg_slots[k].latest = gNeuralAsync::Ticket();
    }
}

void gNeural::Timestep( REAL time )
{
    if ( sn_GetNetState() == nCLIENT )
    {
        if ( sg_localWorker ) StopLocalWorker();
        return;
    }
    if ( !Active() )
        return;
    bool const socketMode = SocketMode();
    if ( !socketMode )
    {
        AssignSlots();
        RefreshLocalLives();
        // Results are polled on every world/main-loop tick, before the 20 Hz observation gate.
        if ( time >= 0 && ( !sg_roundOver || sg_controlAfterRound ) ) ApplyLocalResults( time );
    }
    if ( time < 0 || ( sg_roundOver && !sg_controlAfterRound ) || time + 1E-5 < sg_nextDecision )
        return;
    sg_nextDecision = ( floor( time / sg_interval + 1E-3 ) + 1 ) * sg_interval;

    bool const teacherMode = socketMode && sg_teacher;
    if ( teacherMode && sg_numSlots != 1 )
        Fail( "teacher capture requires one selected slot to avoid sequential-action information leakage" );
    if ( socketMode )
        Connect();
    AssignSlots();

    // out of time: everyone still alive goes down at once, a draw (and nobody's kill)
    bool timeUp = sg_roundTimeLimit > 0 && time > sg_roundTimeLimit && !sg_roundOver;
    if ( timeUp )
        for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
        {
            gCycle * c = dynamic_cast< gCycle * >( se_PlayerNetIDs( i )->Object() );
            if ( c && c->Alive() )
                c->Kill();
        }

    // census of all cycles: who is alive, who died since the last decision and who killed them
    std::vector< int > kills( sg_slots.size(), 0 );
    std::set< void const * > teamsAlive;
    int nAlive = 0;
    for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
    {
        ePlayerNetID * p = se_PlayerNetIDs( i );
        gCycle * c = dynamic_cast< gCycle * >( p->Object() );
        if ( !c )
            continue;
        bool alive = c->Alive();
        if ( alive )
        {
            ++nAlive;
            teamsAlive.insert( c->Team() ? static_cast< void const * >( c->Team() ) : static_cast< void const * >( c ) );
        }
        std::map< ePlayerNetID const *, bool >::iterator it = sg_wasAlive.find( p );
        if ( it != sg_wasAlive.end() && it->second && !alive && !timeUp )
        {
            // a teamkill earns nothing, as in gCycle::KillAt
            ePlayerNetID const * hunter = c->Hunter();
            if ( hunter && hunter != p && hunter->CurrentTeam() != p->CurrentTeam() )
                for ( size_t k = 0; k < sg_slots.size(); ++k )
                    if ( sg_slots[k].player == hunter )
                        ++kills[k];
        }
        sg_wasAlive[ p ] = alive;
    }
    if ( !sg_roundStarted )
    {
        sg_roundStarted = true;
        sg_roundTotal = nAlive;
    }
    bool over = sg_roundOver || ( sg_roundTotal > 1 ? teamsAlive.size() <= 1 : nAlive == 0 );

    if ( sg_debug > 0 && ( sg_tick < 3 || over || sg_debug > 1 ) )
    {
        std::cerr << "neural round " << sg_roundID << " t=" << time << " alive=" << nAlive << " teams="
                  << teamsAlive.size() << " total=" << sg_roundTotal << ( over ? " OVER" : "" ) << "\n";
        for ( int i = 0; i < se_PlayerNetIDs.Len(); ++i )
        {
            ePlayerNetID * p = se_PlayerNetIDs( i );
            gCycle * c = dynamic_cast< gCycle * >( p->Object() );
            std::cerr << "   " << p->GetLogName() << " ai=" << ( dynamic_cast< gAIPlayer * >( p ) != NULL )
                      << " neural=" << ( dynamic_cast< gAIPlayer * >( p ) && gNeural::Controls( dynamic_cast< gAIPlayer * >( p ) ) )
                      << " team=" << ( p->CurrentTeam() ? p->CurrentTeam()->GetLogName() : tString( "-" ) )
                      << " cycle=" << ( c != NULL ) << " cteam=" << ( c ? (void *)c->Team() : NULL )
                      << " alive=" << ( c && c->Alive() );
            if ( c )
                std::cerr << " pos=" << c->Position().x << "," << c->Position().y << " dir=" << c->Direction().x << "," << c->Direction().y;
            std::cerr << "\n";
        }
    }

    // anything to tell the brain?
    bool report = false, slotAlive = false;
    for ( size_t k = 0; k < sg_slots.size(); ++k )
    {
        gCycle * c = SlotCycle( k );
        bool alive = c && c->Alive();
        slotAlive = slotAlive || alive;
        report = report || alive || sg_slots[k].wasAlive || kills[k] > 0;
    }

    if ( sg_spectate && socketMode )
    {
        // WORLD: arena bounds and every cycle, sent before the STEP (if one follows) so a viewer can
        // draw the match; without a STEP the viewer acknowledges, which also lets it pace the game
        tRectangle const & bounds = eWallRim::GetBounds();
        std::vector< std::pair< ePlayerNetID *, gCycle * > > cycles;
        for ( int i = 0; i < se_PlayerNetIDs.Len(); ++i )
        {
            gCycle * c = dynamic_cast< gCycle * >( se_PlayerNetIDs( i )->Object() );
            if ( c )
                cycles.push_back( std::make_pair( se_PlayerNetIDs( i ), c ) );
        }
        Buffer w;
        w.U32( sg_roundID );
        w.F32( time );
        w.U8( over ? 1 : 0 );
        w.U8( report ? 1 : 0 );
        w.F32( bounds.GetLow().x );
        w.F32( bounds.GetLow().y );
        w.F32( bounds.GetHigh().x );
        w.F32( bounds.GetHigh().y );
        w.U8( uint8_t( std::min< size_t >( cycles.size(), 255 ) ) );
        for ( size_t i = 0; i < cycles.size() && i < 255; ++i )
        {
            ePlayerNetID * p = cycles[i].first;
            gCycle * c = cycles[i].second;
            uint8_t slot = 255;
            for ( size_t k = 0; k < sg_slots.size(); ++k )
                if ( sg_slots[k].player == p )
                    slot = uint8_t( k );
            uint16_t id = p->ID();
            w.Raw( &id, 2 );
            w.U8( slot );
            w.U8( c->Alive() ? 1 : 0 );
            eCoord pos = c->Position(), dir = Unit( c->Direction() );
            w.F32( pos.x );
            w.F32( pos.y );
            w.F32( dir.x );
            w.F32( dir.y );
            w.F32( c->Speed() );
            char name[16];
            memset( name, 0, sizeof( name ) );
            strncpy( name, p->GetLogName().c_str(), sizeof( name ) - 1 );
            w.Raw( name, sizeof( name ) );
        }
        SendMessage( kMsgWorld, w );
        if ( !report )
        {
            uint32_t header[3];
            ReadAll( header, sizeof( header ) );
            if ( header[0] != kMagic || header[2] > 1024 )
                Fail( "malformed acknowledgement" );
            std::vector< uint8_t > ack( header[2] );
            if ( header[2] )
                ReadAll( &ack[0], header[2] );
        }
    }

    if ( report )
    {
        int aliveEnemies = 0, aliveMates = 0;
        World world;
        bool steer = !over || sg_controlAfterRound; // keep driving until the next round if asked to
        bool needWorld = steer && slotAlive;
        if ( needWorld )
            BuildWorld( world, time );

        Buffer msg;
        msg.U32( sg_roundID );
        msg.U32( ++sg_tick );
        msg.F32( time );
        msg.U8( over ? 1 : 0 );
        msg.U8( uint8_t( std::min( nAlive, 255 ) ) );
        msg.U8( uint8_t( std::min( sg_roundTotal, 255 ) ) );
        msg.U8( uint8_t( sg_slots.size() ) );

        std::vector< bool > asked( sg_slots.size(), false );
        std::vector< uint8_t > masks( sg_slots.size(), 0 );
        std::vector< size_t > labelOffsets( sg_slots.size(), 0 );
        std::vector< uint8_t > local( kGrid * kGrid ), close( kGrid * kGrid ), global( kGrid * kGrid ), territory( kGrid * kGrid );
        std::vector< float > scalars( kScalars );
        for ( size_t k = 0; k < sg_slots.size(); ++k )
        {
            gCycle * c = SlotCycle( k );
            bool alive = c && c->Alive();
            uint8_t flags = 0;
            if ( alive )
                flags |= FLAG_ALIVE;
            if ( sg_slots[k].wasAlive && !alive )
                flags |= FLAG_DIED;
            if ( over && alive )
                flags |= FLAG_WON;
            asked[k] = alive && steer;
            if ( asked[k] )
                flags |= FLAG_NEEDS_ACTION;
            msg.U8( flags );
            msg.U8( uint8_t( std::min( kills[k], 255 ) ) );
            masks[k] = asked[k] ? ActionMask( c ) : 0;
            msg.U8( masks[k] );
            labelOffsets[k] = msg.data.size();
            msg.U8( 0 );
            if ( asked[k] )
            {
                aliveEnemies = aliveMates = 0;
                for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
                {
                    gCycle * o = dynamic_cast< gCycle * >( se_PlayerNetIDs( i )->Object() );
                    if ( !o || o == c || !o->Alive() )
                        continue;
                    if ( c->Team() && o->Team() == c->Team() )
                        ++aliveMates;
                    else
                        ++aliveEnemies;
                }
                BuildObservation( world, c, time, aliveEnemies, aliveMates, &local[0], &close[0], &global[0], &territory[0], &scalars[0] );
                if ( socketMode )
                {
                    msg.Raw( &local[0], local.size() );
                    msg.Raw( &close[0], close.size() );
                    msg.Raw( &global[0], global.size() );
                    msg.Raw( &territory[0], territory.size() );
                    msg.Raw( &scalars[0], scalars.size() * sizeof( float ) );
                }
                else
                {
                    // Copy only plain data. Inference/memory/RNG run on the worker; its result
                    // is applied on a later main-loop tick after life and legality checks.
                    gNeuralAsync::Observation observation;
                    observation.ticket = { sg_weightGeneration, sg_slots[k].life, sg_tick,
                                           sg_roundID, uint32_t( sg_slots[k].player->ID() ), k };
                    observation.mask = masks[k];
                    observation.sample = sg_sample;
                    observation.gameTime = time;
                    observation.context = sg_slots[k].context.Observe( time );
                    observation.maps.reserve( 4 * local.size() );
                    observation.maps.insert( observation.maps.end(), local.begin(), local.end() );
                    observation.maps.insert( observation.maps.end(), close.begin(), close.end() );
                    observation.maps.insert( observation.maps.end(), global.begin(), global.end() );
                    observation.maps.insert( observation.maps.end(), territory.begin(), territory.end() );
                    observation.scalars = scalars;
                    sg_slots[k].latest = observation.ticket;
                    sg_localWorker->Submit( std::move( observation ) );
                }
            }
            sg_slots[k].wasAlive = alive;
        }
        if ( socketMode )
        {
            // Finish all observations first: earlier teachers must not alter a later slot's
            // observation. The reserved byte is action+1; zero retains normal-mode semantics.
            if ( teacherMode )
                for ( size_t k = 0; k < sg_slots.size(); ++k )
                    if ( asked[k] )
                        msg.data[labelOffsets[k]] = TeacherAction( k, time, masks[k] ) + 1;
            SendMessage( kMsgStep, msg );

            // wait for the decisions
            uint32_t header[3];
            ReadAll( header, sizeof( header ) );
            if ( header[0] != kMagic || header[1] != kMsgActions || header[2] > 1024 )
                Fail( "malformed reply" );
            std::vector< uint8_t > reply( header[2] );
            if ( header[2] )
                ReadAll( &reply[0], header[2] );
            size_t n = reply.empty() ? 0 : reply[0];
            for ( size_t k = 0; k < sg_slots.size() && k < n && k + 1 < reply.size(); ++k )
                if ( asked[k] && !teacherMode ) // teachers already acted; the reply only acknowledges
                    Apply( SlotCycle( k ), reply[ k + 1 ] );
        }
    }

    if ( over )
    {
        sg_roundOver = true;
        if ( sg_localWorker && !sg_controlAfterRound ) sg_localWorker->InvalidateAll();
    }
    else if ( sg_endRound && !slotAlive )
    {
        // nobody left to learn from in this round; end it instead of simulating the rest
        for ( int i = se_PlayerNetIDs.Len() - 1; i >= 0; --i )
        {
            gCycle * o = dynamic_cast< gCycle * >( se_PlayerNetIDs( i )->Object() );
            if ( o && o->Alive() )
                o->Kill();
        }
        sg_roundOver = true;
    }
}
