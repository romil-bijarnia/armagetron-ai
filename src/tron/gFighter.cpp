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

#include "gFighter.h"
#include "../../ai/sim/fighter.h"

#include "gAIBase.h"
#include "gCycle.h"
#include "gCycleMovement.h"
#include "gNeural.h"
#include "gWall.h"
#include "eAdvWall.h"
#include "ePlayer.h"
#include "eTeam.h"
#include "eTimer.h"
#include "nNetwork.h"
#include "tConfiguration.h"
#include "tConsole.h"
#include "tRectangle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

// The fighter is off by default so the training servers and the ladder keep the classic bots
// they measure against; a local preset turns it on.
static bool sg_fighter = false;
static tSettingItem< bool > sg_fighterConf( "AI_FIGHTER", sg_fighter );
static int sg_fighterDepth = 2;      // plies of look-ahead over the real physics (1 or 2)
static tSettingItem< int > sg_fighterDepthConf( "AI_FIGHTER_DEPTH", sg_fighterDepth );
static int sg_fighterPly = 4;        // decisions per ply (0.2 s: a turn and the drive after it)
static tSettingItem< int > sg_fighterPlyConf( "AI_FIGHTER_PLY", sg_fighterPly );
static int sg_fighterProbe = 6;      // decisions it drives straight on past the last ply
static tSettingItem< int > sg_fighterProbeConf( "AI_FIGHTER_PROBE", sg_fighterProbe );
static int sg_fighterThreads = 2;    // worker threads shared by all fighter bots
static tSettingItem< int > sg_fighterThreadsConf( "AI_FIGHTER_THREADS", sg_fighterThreads );
static REAL sg_fighterInterval = .1; // seconds between a bot's decisions; it holds its move in between
static tSettingItem< REAL > sg_fighterIntervalConf( "AI_FIGHTER_INTERVAL", sg_fighterInterval );
static int sg_fighterCheck = 0;      // 1: report how far the simulator drifts from the game, for testing
static tSettingItem< int > sg_fighterCheckConf( "AI_FIGHTER_CHECK", sg_fighterCheck );

namespace
{
REAL const kDecision = .05f; // the simulator's decision interval

struct Job
{
    uint64_t id = 0;
    tron::World world;
    int self = 0;
    tron::FighterConfig cfg;
};

struct Done
{
    uint64_t id = 0;
    int action = tron::ACT_STRAIGHT;
    double ms = 0;
};

// Worker threads see only copied worlds: no game objects, settings or rendering.
class Pool
{
public:
    ~Pool() { Stop(); }

    void Start( int threads )
    {
        std::lock_guard< std::mutex > lock( mutex_ );
        if ( !threads_.empty() ) return;
        stop_ = false;
        for ( int i = 0; i < std::max( 1, threads ); ++i )
            threads_.emplace_back( [this] { Run(); } );
    }

    void Stop()
    {
        {
            std::lock_guard< std::mutex > lock( mutex_ );
            stop_ = true;
        }
        wake_.notify_all();
        for ( std::thread & t : threads_ )
            if ( t.joinable() ) t.join();
        threads_.clear();
        std::lock_guard< std::mutex > lock( mutex_ );
        jobs_.clear();
        done_.clear();
    }

    void Submit( Job && job )
    {
        {
            std::lock_guard< std::mutex > lock( mutex_ );
            jobs_.push_back( std::move( job ) );
        }
        wake_.notify_one();
    }

    std::vector< Done > Poll()
    {
        std::lock_guard< std::mutex > lock( mutex_ );
        std::vector< Done > out;
        out.swap( done_ );
        return out;
    }

    bool Running() const { return !threads_.empty(); }

private:
    void Run()
    {
        for ( ;; )
        {
            Job job;
            {
                std::unique_lock< std::mutex > lock( mutex_ );
                wake_.wait( lock, [this] { return stop_ || !jobs_.empty(); } );
                if ( stop_ ) return;
                job = std::move( jobs_.front() );
                jobs_.pop_front();
            }
            auto start = std::chrono::steady_clock::now();
            Done done;
            done.id = job.id;
            done.action = tron::FighterMove( job.world, job.self, job.cfg );
            done.ms = std::chrono::duration< double, std::milli >( std::chrono::steady_clock::now() - start ).count();
            std::lock_guard< std::mutex > lock( mutex_ );
            done_.push_back( done );
        }
    }

    std::vector< std::thread > threads_;
    std::deque< Job > jobs_;
    std::vector< Done > done_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
};

Pool sg_pool;
uint64_t sg_nextJob = 1;

//! one fighter bot
struct Seat
{
    uint64_t pending = 0;        // the job whose result is awaited
    int planned = -1;            // the result, applied at the next decision
    int last = tron::ACT_STRAIGHT; // the move applied last
    REAL nextDecision = 0;       // game time of the next decision
    REAL nextPlan = 0;           // game time of the next look-ahead
    // the simulator's forecast of this bot for the next decision, when checking
    bool forecast = false, turned = false; // turned: a turn was executed since the forecast
    REAL forecastX = 0, forecastY = 0, forecastDX = 0, forecastDY = 0, forecastSpeed = 0, forecastAt = 0;
};

std::map< gAIPlayer const *, Seat > sg_seats;
double sg_driftSum = 0, sg_driftMax = 0, sg_planMs = 0, sg_planMsMax = 0;
int sg_driftCount = 0, sg_planCount = 0, sg_lateResults = 0, sg_turnSamples = 0;
double sg_turnDriftSum = 0, sg_turnDriftMax = 0;

tron::FighterConfig Config()
{
    tron::FighterConfig cfg;
    cfg.depth = std::max( 1, std::min( 2, sg_fighterDepth ) );
    cfg.ply = std::max( 1, std::min( 10, sg_fighterPly ) );
    cfg.probe = std::max( 0, std::min( 20, sg_fighterProbe ) );
    return cfg;
}

void Physics( tron::Settings & s )
{
    gCyclePhysics ph = gCycleMovement::ExportPhysics();
    s.speed = ph.speed;
    s.speedMin = ph.speedMin;
    s.speedMax = ph.speedMax;
    s.decayBelow = ph.decayBelow;
    s.decayAbove = ph.decayAbove;
    s.startSpeed = ph.startSpeed;
    s.accel = ph.accel;
    s.accelOffset = ph.accelOffset;
    s.wallNear = ph.wallNear;
    s.accelSelf = ph.accelSelf;
    s.accelTeam = ph.accelTeam;
    s.accelEnemy = ph.accelEnemy;
    s.accelRim = ph.accelRim;
    s.accelSlingshot = ph.accelSlingshot;
    s.accelTunnel = ph.accelTunnel;
    s.brake = ph.brake;
    s.brakeRefill = ph.brakeRefill;
    s.brakeDeplete = ph.brakeDeplete;
    s.delay = ph.delay;
    s.delayBonus = ph.delayBonus;
    s.turnSpeedFactor = ph.turnSpeedFactor;
    s.rubber = ph.rubber;
    s.rubberSpeed = ph.rubberSpeed;
    s.rubberTime = ph.rubberTime;
    s.rubberMinDistance = ph.rubberMinDistance;
    s.rubberMinDistanceRatio = ph.rubberMinDistanceRatio;
    s.rubberMinDistanceReservoir = ph.rubberMinDistanceReservoir;
    s.rubberMinDistanceUnprepared = ph.rubberMinDistanceUnprepared;
    s.rubberMinDistancePreparation = ph.rubberMinDistancePreparation;
    s.rubberMinAdjust = ph.rubberMinAdjust;
    s.turnRadius = ph.turnRadius;
    s.turnRadiusMin = ph.turnRadiusMin;
    s.turnWallStep = ph.turnWallStep;
    s.wallsLength = gCycle::WallsLength() > 0 ? gCycle::WallsLength() : 1e9f;
    s.wallsStayUp = gCycle::WallsStayUpDelay() >= 0 ? gCycle::WallsStayUpDelay() : 1e9f;
    s.explosionRadius = gCycle::ExplosionRadius();
    // a dedicated server draws each wall a little ahead of its cycle for lag; a local game does not
    s.predictAhead = sn_GetNetState() == nSERVER ? .175f : 0;
}

struct WallRun
{
    gNetPlayerWall * wall;
    REAL begin;
};

void CollectWalls( tList< gNetPlayerWall > & list, std::map< gCycle const *, std::vector< WallRun > > & walls )
{
    for ( int i = list.Len() - 1; i >= 0; --i )
    {
        gNetPlayerWall * wall = list( i );
        if ( !wall || !wall->Cycle() )
            continue;
        WallRun run = { wall, wall->BegPos() };
        walls[ wall->Cycle() ].push_back( run );
    }
}

//! the trail of one cycle, oldest turn first, from its walls
void Trail( gCycle const * cycle, std::vector< WallRun > & runs, tron::Cycle & c )
{
    std::sort( runs.begin(), runs.end(), []( WallRun const & a, WallRun const & b ) { return a.begin < b.begin; } );
    c.trail.clear();
    c.holes.clear();
    for ( size_t i = 0; i < runs.size(); ++i )
    {
        gNetPlayerWall * wall = runs[i].wall;
        eCoord const & a = wall->EndPoint( 0 );
        if ( c.trail.empty() || wall->BegPos() > c.trail.back().dist + 1e-3f )
            c.trail.push_back( tron::Coord{ a.x, a.y, wall->BegPos(), wall->BegTime() } );
        tArray< gPlayerWallCoord > & coords = wall->Coords();
        for ( int k = 0; k + 1 < coords.Len(); ++k )
            if ( !coords( k ).IsDangerous )
                c.holes.push_back( std::make_pair( coords( k ).Pos, coords( k + 1 ).Pos ) );
        if ( i + 1 == runs.size() )
        {
            eCoord const & b = wall->EndPoint( 1 );
            c.predX = b.x;
            c.predY = b.y;
        }
    }
    if ( c.trail.empty() )
    {
        eCoord const & start = cycle->GetLastTurnPos();
        c.trail.push_back( tron::Coord{ start.x, start.y, 0, cycle->LastTime() } );
        c.predX = cycle->Position().x;
        c.predY = cycle->Position().y;
    }
}

//! the arena as the simulator sees it; SELF becomes cycle 0; false if nothing is to be done
bool Snapshot( tron::World & w, gCycle * self, REAL time )
{
    tRectangle const & bounds = eWallRim::GetBounds();
    w.minX = bounds.GetLow().x;
    w.minY = bounds.GetLow().y;
    w.maxX = bounds.GetHigh().x;
    w.maxY = bounds.GetHigh().y;
    w.s = tron::Settings();
    Physics( w.s );
    w.time = time;

    // the cycles, nearest to the fighter first; the simulator seats at most kMaxPlayers
    std::vector< gCycle * > cycles;
    for ( int i = 0; i < se_PlayerNetIDs.Len(); ++i )
    {
        gCycle * c = dynamic_cast< gCycle * >( se_PlayerNetIDs( i )->Object() );
        if ( c && c != self )
            cycles.push_back( c );
    }
    eCoord const origin = self->Position();
    std::sort( cycles.begin(), cycles.end(), [&origin]( gCycle * a, gCycle * b )
               { return ( a->Position() - origin ).NormSquared() < ( b->Position() - origin ).NormSquared(); } );
    cycles.insert( cycles.begin(), self );
    if ( int( cycles.size() ) > tron::kMaxPlayers )
        cycles.resize( tron::kMaxPlayers );

    std::map< gCycle const *, std::vector< WallRun > > walls;
    CollectWalls( sg_netPlayerWalls, walls );
    CollectWalls( sg_netPlayerWallsGridded, walls );

    std::map< void const *, int > teams;
    w.cycles.assign( cycles.size(), tron::Cycle() );
    for ( size_t i = 0; i < cycles.size(); ++i )
    {
        gCycle * g = cycles[i];
        tron::Cycle & c = w.cycles[i];
        void const * team = g->Team() ? static_cast< void const * >( g->Team() ) : static_cast< void const * >( g );
        if ( !teams.count( team ) )
            teams[ team ] = int( teams.size() );
        c.team = teams[ team ];
        c.alive = g->Alive();
        c.deathTime = g->DeathTime();
        c.expansion = c.alive ? 0 : 5; // the holes of a finished explosion are already in its walls
        c.x = g->Position().x;
        c.y = g->Position().y;
        c.dx = g->Direction().x;
        c.dy = g->Direction().y;
        c.verletSpeed = g->Speed();
        c.acceleration = g->GetAcceleration();
        c.lastTimestep = 0;
        c.braking = g->GetBraking() != 0;
        c.brakingReservoir = g->GetBrakingReservoir();
        c.rubber = g->GetRubber();
        c.rubberMalus = g->GetRubberMalus();
        c.lastTurnLeft = g->GetLastTurnTimeLeft();
        c.lastTurnRight = g->GetLastTurnTimeRight();
        c.lastTime = g->LastTime(); // a server simulates cycles a little ahead of the clock
        c.distance = g->GetDistance();
        c.lastTurnX = g->GetLastTurnPos().x;
        c.lastTurnY = g->GetLastTurnPos().y;
        gSweep sweep = g->GetSweep();
        c.arcActive = sweep.active;
        c.arcHeading = sweep.heading;
        c.arcTarget = sweep.target;
        c.arcRadius = sweep.radius;
        c.arcDropped = sweep.dropped;
        c.axisDX = sweep.axis.x;
        c.axisDY = sweep.axis.y;
        c.arcDropX = sweep.dropPos.x;
        c.arcDropY = sweep.dropPos.y;
        Trail( g, walls[ g ], c );
    }
    w.roundTotal = int( cycles.size() );
    return !w.cycles.empty() && w.cycles[0].alive;
}

void Report()
{
    if ( sg_fighterCheck <= 0 || sg_planCount == 0 )
        return;
    std::cerr << "fighter check: t=" << se_GameTime() << " plans=" << sg_planCount << " ms=" << sg_planMs / sg_planCount
              << " max=" << sg_planMsMax << " late=" << sg_lateResults;
    if ( sg_driftCount > 0 )
        std::cerr << " drift(m)=" << sg_driftSum / sg_driftCount << " max=" << sg_driftMax
                  << " samples=" << sg_driftCount << " turns=" << sg_turnSamples;
    if ( sg_turnSamples > 0 )
        std::cerr << " turnDrift(m)=" << sg_turnDriftSum / sg_turnSamples << " max=" << sg_turnDriftMax;
    std::cerr << "\n";
}
}

bool gFighter::Controls( gAIPlayer const * player )
{
    if ( !sg_fighter || !player || sn_GetNetState() == nCLIENT )
        return false;
    return !gNeural::Controls( player );
}

void gFighter::Timestep( gAIPlayer * player, REAL time )
{
    gCycle * cycle = dynamic_cast< gCycle * >( player->Object() );
    if ( !cycle || !cycle->Alive() || time < 0 )
        return;
    if ( !sg_pool.Running() )
        sg_pool.Start( sg_fighterThreads );

    // results of the look-aheads, for whichever bot asked
    for ( Done const & done : sg_pool.Poll() )
    {
        sg_planMs += done.ms;
        sg_planMsMax = std::max( sg_planMsMax, done.ms );
        ++sg_planCount;
        for ( std::map< gAIPlayer const *, Seat >::iterator it = sg_seats.begin(); it != sg_seats.end(); ++it )
            if ( it->second.pending == done.id )
            {
                it->second.pending = 0;
                it->second.planned = done.action;
            }
    }

    Seat & seat = sg_seats[ player ];
    if ( time + 1e-5f < seat.nextDecision )
        return;
    seat.nextDecision = ( floor( time / kDecision + 1e-3 ) + 1 ) * kDecision;

    // how far the forecast of the last decision was off: the simulator against the game
    if ( sg_fighterCheck > 0 && seat.forecast && time >= seat.forecastAt - 1e-4f )
    {
        // the cycle's own clock is a little past the decision time: carry the forecast on to it
        REAL ahead = std::max( 0.f, cycle->LastTime() - seat.forecastAt ) * seat.forecastSpeed;
        REAL dx = cycle->Position().x - ( seat.forecastX + seat.forecastDX * ahead );
        REAL dy = cycle->Position().y - ( seat.forecastY + seat.forecastDY * ahead );
        REAL drift = sqrt( dx * dx + dy * dy );
        // a turn executed since the forecast moves the cycle off the forecast's line by design
        if ( seat.turned )
        {
            ++sg_turnSamples;
            sg_turnDriftSum += drift;
            sg_turnDriftMax = std::max< double >( sg_turnDriftMax, drift );
        }
        else
        {
            sg_driftSum += drift;
            sg_driftMax = std::max< double >( sg_driftMax, drift );
            ++sg_driftCount;
        }
        seat.forecast = false;
        if ( ( sg_driftCount + sg_turnSamples ) % 100 == 0 )
            Report();
    }

    // the move for this decision: the look-ahead's answer if it arrived, else hold the last one
    int action = tron::Hold( seat.last );
    if ( seat.planned >= 0 )
    {
        action = seat.planned;
        seat.planned = -1;
    }
    else if ( seat.pending )
        ++sg_lateResults;
    unsigned legal = gNeural::LegalActions( cycle );
    if ( !( legal >> action & 1 ) )
        action = tron::Hold( action );
    int executed = gNeural::ApplyAction( cycle, action );
    seat.last = executed >= 0 ? executed : action;
    if ( sg_fighterCheck > 1 )
        std::cerr << "fighter " << ( player->GetName() ) << " t=" << time << " pos=" << cycle->Position().x << ","
                  << cycle->Position().y << " dir=" << cycle->Direction().x << "," << cycle->Direction().y
                  << " speed=" << cycle->Speed() << " act=" << action << " exec=" << executed
                  << " sweep=" << cycle->Sweeping() << " legal=" << legal << " turns=" << cycle->GetTurns()
                  << " pending=" << cycle->PendingTurnCount()
                  << " lastTurn=" << cycle->GetLastTurnTime() << " own=" << cycle->LastTime()
                  << " h=" << cycle->GetSweep().heading << " tg=" << cycle->GetSweep().target
                  << " ax=" << cycle->GetSweep().axis.x << "," << cycle->GetSweep().axis.y << "\n";
    if ( seat.last != tron::ACT_STRAIGHT && seat.last != tron::ACT_BRAKE )
        seat.turned = true;

    if ( time + 1e-5f < seat.nextPlan || seat.pending )
        return;
    seat.nextPlan = time + std::max( kDecision, sg_fighterInterval ) - 1e-4f;

    // the look-ahead starts from where the arena will be at the next decision: this move holds
    // for the bot, everyone else drives on
    Job job;
    job.id = sg_nextJob++;
    job.cfg = Config();
    if ( !Snapshot( job.world, cycle, time ) )
        return;
    int acts[ tron::kMaxPlayers ] = { 0 };
    for ( size_t i = 0; i < job.world.cycles.size(); ++i )
        acts[i] = job.world.cycles[i].braking ? tron::ACT_BRAKE : tron::ACT_STRAIGHT;
    acts[0] = tron::Hold( seat.last );
    job.world.Step( acts );
    if ( sg_fighterCheck > 0 )
    {
        seat.forecast = true;
        seat.turned = false;
        seat.forecastX = job.world.cycles[0].x;
        seat.forecastY = job.world.cycles[0].y;
        seat.forecastDX = job.world.cycles[0].dx;
        seat.forecastDY = job.world.cycles[0].dy;
        seat.forecastSpeed = job.world.cycles[0].Speed();
        seat.forecastAt = job.world.cycles[0].lastTime; // the forecast cycle's own clock
    }
    if ( !job.world.cycles[0].alive )
        return;
    seat.pending = job.id;
    sg_pool.Submit( std::move( job ) );
}

void gFighter::NewRound()
{
    sg_seats.clear();
    Report();
}

void gFighter::Shutdown()
{
    sg_pool.Stop();
    sg_seats.clear();
}
