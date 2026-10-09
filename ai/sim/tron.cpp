// Cycle physics, walls and explosions: a port of the engine's code paths that a dedicated server
// with AI players goes through. Function and variable names follow the engine's so the two can be
// read side by side (gCycleMovement.cpp: TimestepCore, CalculateAcceleration, ApplyAcceleration,
// GetMaxSpaceAhead, DoTurn; gCycle.cpp: PassEdge, KillAt; gWall.cpp: IsDangerous; gExplosion.cpp).

#include "tron.h"

#include <algorithm>
#include <cmath>

namespace tron
{
namespace
{
// square-1.0.1.aamap.xml, scaled by the arena size
const REAL kSpawns[12][4] = {
    { 255, 50, 0, 1 },   { 245, 450, 0, -1 }, { 50, 245, 1, 0 },   { 450, 255, -1, 0 },
    { 305, 100, 0, 1 },  { 195, 400, 0, -1 }, { 100, 195, 1, 0 },  { 400, 305, -1, 0 },
    { 205, 100, 0, 1 },  { 295, 400, 0, -1 }, { 100, 295, 1, 0 },  { 400, 205, -1, 0 } };

inline bool ClampTo( REAL & v, REAL lo, REAL hi )
{
    if ( v < lo ) { v = lo; return true; }
    if ( v > hi ) { v = hi; return true; }
    return false;
}

struct Guard
{
    bool & flag;
    explicit Guard( bool & f ): flag( f ) { flag = false; }
    ~Guard() { flag = true; }
};

// the engine's static recursion guards (one per split reason)
thread_local bool sg_recurseBrake = true, sg_recurseRubberStart = true, sg_recurseObstacle = true, sg_recurseRunOut = true;
}

// ------------------------------------------------------------------ setup
void World::Reset( REAL sizeFactor, int n, int rotate, int const * teams )
{
    // gArena: the map is scaled by 2^(SIZE_FACTOR/2)
    REAL scale = REAL( std::exp( double( sizeFactor ) * .5 * std::log( 2.0 ) ) );
    minX = minY = 0;
    maxX = maxY = 500 * scale;
    time = 0;
    cycles.assign( n, Cycle() );
    roundTotal = n;
    for ( int i = 0; i < n; ++i )
    {
        int k = ( ( i + rotate ) % n + n ) % n % 12;
        Cycle & c = cycles[i];
        c.team = teams ? teams[i] : i;
        Place( i, kSpawns[k][0] * scale, kSpawns[k][1] * scale, kSpawns[k][2], kSpawns[k][3] );
    }
}

void World::Place( int i, REAL x, REAL y, REAL dx, REAL dy )
{
    Cycle & c = cycles[i];
    int team = c.team;
    c = Cycle();
    c.team = team;
    c.x = x;
    c.y = y;
    c.dx = dx;
    c.dy = dy;
    c.verletSpeed = s.startSpeed;
    c.lastTime = time;
    c.lastTurnLeft = c.lastTurnRight = time - 10;
    c.lastTurnX = x;
    c.lastTurnY = y;
    c.predX = x;
    c.predY = y;
    c.trail.push_back( Coord{ x, y, 0, time } );
}

// ------------------------------------------------------------------ walls
int World::NumWalls( int c ) const
{
    return int( cycles[c].trail.size() );
}

//! endpoints and trail distances of wall SEG of cycle C; the current wall ends at the cycle, or at
//! the predicted position when PREDICTED (that is how far the engine draws it)
void World::WallEnds( int c, int seg, REAL & x0, REAL & y0, REAL & d0, REAL & x1, REAL & y1, REAL & d1, bool predicted ) const
{
    Cycle const & cy = cycles[c];
    Coord const & a = cy.trail[seg];
    x0 = a.x;
    y0 = a.y;
    d0 = a.dist;
    if ( seg + 1 < int( cy.trail.size() ) )
    {
        Coord const & b = cy.trail[seg + 1];
        x1 = b.x;
        y1 = b.y;
        d1 = b.dist;
    }
    else if ( predicted && cy.alive )
    {
        x1 = cy.predX;
        y1 = cy.predY;
        d1 = cy.distance + std::fabs( cy.predX - cy.x ) + std::fabs( cy.predY - cy.y );
    }
    else
    {
        x1 = cy.x;
        y1 = cy.y;
        d1 = cy.distance;
    }
}

//! when the wall at trail distance W was built (linear along each wall; extrapolated past the cycle)
REAL World::WallTimeAt( int c, int seg, REAL w ) const
{
    Cycle const & cy = cycles[c];
    Coord const & a = cy.trail[seg];
    REAL d1, t1;
    if ( seg + 1 < int( cy.trail.size() ) )
    {
        d1 = cy.trail[seg + 1].dist;
        t1 = cy.trail[seg + 1].time;
    }
    else
    {
        d1 = cy.distance;
        t1 = cy.lastTime;
        if ( w > d1 )
        {
            REAL v = cy.verletSpeed > .1f ? cy.verletSpeed : .1f;
            return t1 + ( w - d1 ) / v;
        }
    }
    if ( d1 <= a.dist )
        return t1;
    return a.time + ( t1 - a.time ) * ( w - a.dist ) / ( d1 - a.dist );
}

bool World::WallPointDangerous( int ci, REAL w, REAL t ) const
{
    Cycle const & c = cycles[ci];
    // walls disappear after death
    if ( !c.alive && c.deathTime + s.wallsStayUp + .2f <= t )
        return false;
    REAL dt = t - c.lastTime;
    // finite trails: the far end recedes
    if ( s.wallsLength > 0 )
    {
        REAL cycleDistance = c.distance;
        if ( c.alive )
            cycleDistance += c.rubberSpeedFactor * c.Speed() * dt;  // WallEndSpeed
        if ( w + s.wallsLength < cycleDistance )
            return false;
    }
    // the bit ahead of the cycle is only a prediction
    {
        REAL cycleDistance = c.distance;
        if ( c.alive && dt > 0 )
            cycleDistance += c.Speed() * c.rubberSpeedFactor * dt;
        if ( w > cycleDistance )
            return false;
    }
    for ( size_t h = 0; h < c.holes.size(); ++h )
        if ( w >= c.holes[h].first && w <= c.holes[h].second )
            return false;
    return true;
}

//! gCycle::EdgeIsDangerous for cycle SELF looking at wall SEG of cycle OWNER
bool World::EdgeDangerousFor( int self, int owner, int seg, REAL w, REAL t ) const
{
    if ( self >= 0 && self == owner )
    {
        // a cycle never collides with the wall it is building nor with the one before it
        int n = NumWalls( owner );
        if ( cycles[owner].alive && ( seg == n - 1 || seg == n - 2 ) )
            return false;
    }
    return WallPointDangerous( owner, w, t );
}

// ------------------------------------------------------------------ sensors
Hit World::Cast( int owner, REAL px, REAL py, REAL ddx, REAL ddy, REAL range, REAL inverseSpeed ) const
{
    Hit best;
    best.t = range + .00001f;
    REAL const tmin = 1E-6f;
    REAL baseTime = owner >= 0 ? cycles[owner].lastTime : time;

    // the rim
    {
        REAL const xs[2] = { minX, maxX }, ys[2] = { minY, maxY };
        for ( int k = 0; k < 2; ++k )
        {
            if ( ddx != 0 )
            {
                REAL t = ( xs[k] - px ) / ddx;
                if ( t > tmin && t < best.t )
                {
                    REAL y = py + ddy * t;
                    if ( y >= minY && y <= maxY )
                    {
                        best.t = t; best.type = S_RIM; best.owner = -1; best.seg = -1;
                        best.wallLen = maxY - minY; best.x = xs[k]; best.y = y;
                    }
                }
            }
            if ( ddy != 0 )
            {
                REAL t = ( ys[k] - py ) / ddy;
                if ( t > tmin && t < best.t )
                {
                    REAL x = px + ddx * t;
                    if ( x >= minX && x <= maxX )
                    {
                        best.t = t; best.type = S_RIM; best.owner = -1; best.seg = -1;
                        best.wallLen = maxX - minX; best.x = x; best.y = ys[k];
                    }
                }
            }
        }
    }

    // trails
    for ( int c = 0; c < int( cycles.size() ); ++c )
    {
        Cycle const & cy = cycles[c];
        if ( !cy.alive && cy.deathTime + s.wallsStayUp + .2f <= baseTime )
            continue;
        int n = NumWalls( c );
        for ( int seg = 0; seg < n; ++seg )
        {
            REAL x0, y0, d0, x1, y1, d1;
            WallEnds( c, seg, x0, y0, d0, x1, y1, d1, true );
            REAL t, hx, hy, along;
            if ( x0 == x1 )
            {
                if ( y0 == y1 || ddx == 0 )
                    continue;
                t = ( x0 - px ) / ddx;
                if ( t <= tmin || t >= best.t )
                    continue;
                hy = py + ddy * t;
                if ( hy < std::min( y0, y1 ) || hy > std::max( y0, y1 ) )
                    continue;
                hx = x0;
                along = std::fabs( hy - y0 );
            }
            else if ( y0 == y1 )
            {
                if ( ddy == 0 )
                    continue;
                t = ( y0 - py ) / ddy;
                if ( t <= tmin || t >= best.t )
                    continue;
                hx = px + ddx * t;
                if ( hx < std::min( x0, x1 ) || hx > std::max( x0, x1 ) )
                    continue;
                hy = y0;
                along = std::fabs( hx - x0 );
            }
            else
                continue;  // walls are axis-aligned
            REAL w = d0 + along;
            REAL hitTime = baseTime + t * inverseSpeed;
            if ( !EdgeDangerousFor( owner, c, seg, w, hitTime ) )
                continue;
            // sensors (not real moves) ignore other cycles' walls from the future (gJustChecking)
            if ( owner >= 0 && c != owner && WallTimeAt( c, seg, w ) > hitTime )
            {
                bool const sameTeam = cycles[c].team == cycles[owner].team;
                if ( !sameTeam || seg == n - 1 )
                    continue;
            }
            best.t = t;
            best.owner = c;
            best.seg = seg;
            best.wallLen = std::fabs( x1 - x0 ) + std::fabs( y1 - y0 );
            best.x = hx;
            best.y = hy;
            best.wallDist = w;
            if ( owner >= 0 && c == owner )
                best.type = S_SELF;
            else if ( owner >= 0 && cycles[c].team == cycles[owner].team )
                best.type = S_TEAMMATE;
            else
                best.type = S_ENEMY;
        }
    }
    return best;
}

// ------------------------------------------------------------------ turning
REAL World::TurnDelay( int i ) const
{
    // CYCLE_DELAY * CYCLE_DELAY_BONUS * (speed/base)^(CYCLE_DELAY_TIMEBASED - 1), the exponent being 0
    (void)i;
    return s.delay * s.delayBonus;
}

REAL World::NextTurn( int i ) const
{
    Cycle const & c = cycles[i];
    REAL d = TurnDelay( i );
    REAL a = c.lastTurnRight + d, b = c.lastTurnLeft + d;
    return a > b ? a : b;
}

unsigned World::ActionMask( int i ) const
{
    Cycle const & c = cycles[i];
    unsigned mask = ( 1u << ACT_STRAIGHT ) | ( 1u << ACT_BRAKE );
    if ( c.lastTime >= NextTurn( i ) )
        mask |= ( 1u << ACT_LEFT ) | ( 1u << ACT_RIGHT );
    return mask;
}

bool World::DoTurn( int i, int dir )
{
    Cycle & c = cycles[i];
    if ( NextTurn( i ) > c.lastTime )
        return false;  // the engine would queue it; the network's mask never asks for that

    c.refreshSpace = true;
    c.rubberSpeedFactor = 1;
    c.lastTurnX = c.x;
    c.lastTurnY = c.y;

    // AccelerationDiscontinuity
    c.verletSpeed = c.Speed();
    c.lastTimestep = 0;
    c.verletSpeed *= s.turnSpeedFactor;

    // left (dir -1) turns counter-clockwise
    REAL ndx, ndy;
    if ( dir < 0 )
    {
        ndx = -c.dy;
        ndy = c.dx;
    }
    else
    {
        ndx = c.dy;
        ndy = -c.dx;
    }
    if ( dir == 1 )
        c.lastTurnRight = c.lastTime;
    else
        c.lastTurnLeft = c.lastTime;
    c.dx = ndx;
    c.dy = ndy;

    // DropWall: the current wall ends here, a new one starts
    c.trail.push_back( Coord{ c.x, c.y, c.distance, c.lastTime } );
    c.predX = c.x;
    c.predY = c.y;
    return true;
}

void World::Act( int i, int action )
{
    Cycle & c = cycles[i];
    if ( !c.alive )
        return;
    bool brake = action == ACT_BRAKE;
    if ( c.braking != brake )
    {
        c.verletSpeed = c.Speed();  // AccelerationDiscontinuity
        c.lastTimestep = 0;
        c.braking = brake;
    }
    if ( action == ACT_LEFT )
        DoTurn( i, -1 );
    else if ( action == ACT_RIGHT )
        DoTurn( i, 1 );
}

// ------------------------------------------------------------------ acceleration
void World::CalculateAcceleration( int i )
{
    Cycle & c = cycles[i];
    c.brakeUsage = 0;
    c.rubberUsage = 0;
    c.acceleration = 0;

    if ( c.braking )
    {
        if ( c.brakingReservoir > 0 )
        {
            c.brakeUsage = s.brakeDeplete;
            c.acceleration -= s.brake;
        }
        else
            c.brakingReservoir = 0;
    }
    else
    {
        if ( c.brakingReservoir < 1 )
            c.brakeUsage = -s.brakeRefill;
        else
            c.brakingReservoir = 1;
    }

    REAL baseSpeed = s.speed;
    if ( c.verletSpeed <= baseSpeed )
        c.acceleration += ( baseSpeed - c.verletSpeed ) * s.decayBelow;
    else
        c.acceleration += ( baseSpeed - c.verletSpeed ) * s.decayAbove;

    // sense walls diagonally behind: driving close along them speeds the cycle up
    REAL totalWallAcceleration = 0;
    bool slingshot = true, oneOwnWall = false;
    for ( int d = 1; d >= -1; d -= 2 )
    {
        // dirDrive.Turn(-1, d): a complex multiplication, 135 degrees back, length sqrt(2)
        REAL cx = -c.dx - c.dy * d, cy = -c.dy + c.dx * d;
        Hit rear = Cast( i, c.x, c.y, cx, cy, s.wallNear );
        if ( rear.type != S_NONE )
        {
            if ( rear.type == S_ENEMY )
            {
                c.hunter = rear.owner;
                c.hunterTime = c.lastTime;
            }
            // only walls parallel to the driving direction (and longer than .9 m) push
            REAL wx = 0, wy = 0;
            if ( rear.owner >= 0 )
            {
                REAL x0, y0, d0, x1, y1, d1;
                WallEnds( rear.owner, rear.seg, x0, y0, d0, x1, y1, d1, true );
                wx = x1 - x0;
                wy = y1 - y0;
            }
            else
            {
                // rim: horizontal or vertical depending on which side was hit
                bool vertical = rear.x == minX || rear.x == maxX;
                wx = vertical ? 0 : rear.wallLen;
                wy = vertical ? rear.wallLen : 0;
            }
            if ( std::fabs( wx * c.dx + wy * c.dy ) > .9f )
            {
                REAL wallAcceleration = s.accel * ( ( 1 / ( rear.t + s.accelOffset ) ) - ( 1 / ( s.wallNear + s.accelOffset ) ) );
                switch ( rear.type )
                {
                case S_SELF:
                    wallAcceleration *= s.accelSelf;
                    oneOwnWall = true;
                    break;
                case S_TEAMMATE:
                    wallAcceleration *= s.accelTeam;
                    break;
                case S_ENEMY:
                    wallAcceleration *= s.accelEnemy;
                    break;
                case S_RIM:
                    wallAcceleration *= s.accelRim;
                    break;
                }
                totalWallAcceleration += wallAcceleration;
            }
            else
                slingshot = false;
        }
        else
            slingshot = false;
    }
    if ( slingshot )
        totalWallAcceleration *= oneOwnWall ? s.accelSlingshot : s.accelTunnel;
    c.acceleration += totalWallAcceleration;
}

void World::ApplyAcceleration( int i, REAL dt )
{
    Cycle & c = cycles[i];
    REAL verletTimestep = .5f * ( dt + c.lastTimestep );
    c.lastTimestep = dt;

    bool properDecay = false;
    REAL maxTimestep = verletTimestep > dt ? verletTimestep : dt;
    if ( s.decayBelow * maxTimestep > .1f || s.decayAbove * maxTimestep > .1f )
    {
        REAL speedDecay = 0;
        REAL baseSpeed = s.speed;
        if ( c.verletSpeed < baseSpeed )
            speedDecay = s.decayBelow;
        else
            speedDecay = s.decayAbove;

        if ( speedDecay * maxTimestep > .1f && dt > EPS )
        {
            properDecay = true;
            REAL decayAcceleration = ( baseSpeed - c.verletSpeed ) * speedDecay;
            c.acceleration -= decayAcceleration;
            baseSpeed += c.acceleration / speedDecay;
            c.verletSpeed = baseSpeed + ( c.verletSpeed - baseSpeed ) * std::exp( -speedDecay * verletTimestep );
            c.acceleration = ( baseSpeed - c.verletSpeed ) * ( 1 - std::exp( -speedDecay * dt * .5f ) ) / ( .5f * dt );
        }
    }
    if ( !properDecay )
        c.verletSpeed += c.acceleration * verletTimestep;

    REAL minSpeed = s.speed * s.speedMin;
    REAL maxSpeed = ( 100 + s.speed ) * 100000;
    if ( s.speedMax > 0 )
        maxSpeed = s.speed * s.speedMax;
    if ( ClampTo( c.verletSpeed, minSpeed, maxSpeed ) )
        c.acceleration = 0;
}

// ------------------------------------------------------------------ rubber lookahead
REAL World::GetMaxSpaceAhead( int i, REAL maxReport )
{
    Cycle & c = cycles[i];
    if ( c.refreshSpace )
    {
        c.refreshSpace = false;
        REAL lookAhead = c.maxSpaceMaxCast;
        if ( maxReport > lookAhead )
            lookAhead = maxReport;

        REAL mindistance = s.rubberMinDistance;
        {
            REAL rubberGranted = s.rubber, rubberEffectiveness = 1;
            if ( rubberGranted > 0 )
            {
                REAL rubberUsageSpeed = c.verletSpeed * ( 1 - c.rubberSpeedFactor ) / rubberEffectiveness;
                REAL rubberUsed = rubberUsageSpeed * c.lastTimestep;
                REAL filling = ( c.rubber + rubberUsed ) / rubberGranted;
                if ( filling > 1 )
                    filling = 1;
                mindistance += s.rubberMinDistanceReservoir * ( 1 - filling );
            }
            if ( s.rubberMinDistancePreparation > 0 )
            {
                REAL badPreparation = s.rubberMinDistancePreparation /
                                      ( s.rubberMinDistancePreparation + ( c.lastTime - c.LastTurnTime() ) );
                mindistance += s.rubberMinDistanceUnprepared * badPreparation;
            }
        }
        lookAhead += mindistance * 2;  // CYCLE_RUBBER_MINDISTANCE_LEGACY 1

        REAL speed = c.Speed();
        Hit fr = Cast( i, c.x, c.y, c.dx, c.dy, lookAhead, speed > 0 ? 1 / speed : 0 );
        if ( fr.type != S_NONE )
        {
            REAL stopDistance = mindistance + s.rubberMinDistanceRatio * fr.wallLen;
            REAL space = fr.t;
            REAL maxStop = ( c.DistanceSinceLastTurn() + space ) * ( 1 - s.rubberMinAdjust );
            if ( maxStop < stopDistance )
                stopDistance = maxStop;
            REAL safety = std::sqrt( c.x * c.x + c.y * c.y ) * 2 * EPS;
            c.spaceHit = true;
            c.spaceX = fr.x - c.dx * .000001f;
            c.spaceY = fr.y - c.dy * .000001f;
            c.spaceOffset = stopDistance + safety;
            c.spaceOwner = fr.owner;
            c.spaceSeg = fr.seg;
            c.spaceWallDist = fr.wallDist;
            if ( fr.type == S_ENEMY )
            {
                c.hunter = fr.owner;
                c.hunterTime = c.lastTime;
            }
        }
        else
            c.spaceHit = false;
    }
    REAL ret = 1E+30f;
    if ( c.spaceHit )
        ret = ( c.spaceX - c.x ) * c.dx + ( c.spaceY - c.y ) * c.dy - c.spaceOffset;
    if ( ret > maxReport )
        ret = maxReport;
    return ret;
}

// ------------------------------------------------------------------ moving and dying
//! eGameObject::Move with gCycle::PassEdge: drive to (NX, NY), crossing walls on the way
void World::Move( int i, REAL nx, REAL ny, REAL t0, REAL t1 )
{
    Cycle & me = cycles[i];
    REAL sx = me.x, sy = me.y;
    REAL mx = nx - sx, my = ny - sy;
    REAL len = std::fabs( mx ) + std::fabs( my );
    if ( len <= 0 )
        return;

    struct Crossing { REAL u; int owner; int seg; REAL x, y, w; };
    Crossing found[64];
    int nFound = 0;
    REAL const umin = 1E-7f;

    auto consider = [&]( REAL u, int owner, int seg, REAL x, REAL y, REAL w )
    {
        if ( u <= umin || u > 1 || nFound >= 64 )
            return;
        found[nFound++] = Crossing{ u, owner, seg, x, y, w };
    };

    // the rim
    {
        REAL const xs[2] = { minX, maxX }, ys[2] = { minY, maxY };
        for ( int k = 0; k < 2; ++k )
        {
            if ( mx != 0 )
            {
                REAL u = ( xs[k] - sx ) / mx;
                REAL y = sy + my * u;
                if ( y >= minY && y <= maxY )
                    consider( u, -1, -1, xs[k], y, 0 );
            }
            if ( my != 0 )
            {
                REAL u = ( ys[k] - sy ) / my;
                REAL x = sx + mx * u;
                if ( x >= minX && x <= maxX )
                    consider( u, -1, -1, x, ys[k], 0 );
            }
        }
    }
    for ( int c = 0; c < int( cycles.size() ); ++c )
    {
        Cycle const & cy = cycles[c];
        if ( !cy.alive && cy.deathTime + s.wallsStayUp + .2f <= t0 )
            continue;
        int n = NumWalls( c );
        for ( int seg = 0; seg < n; ++seg )
        {
            REAL x0, y0, d0, x1, y1, d1;
            WallEnds( c, seg, x0, y0, d0, x1, y1, d1, true );
            if ( x0 == x1 && y0 != y1 && mx != 0 )
            {
                REAL u = ( x0 - sx ) / mx;
                REAL y = sy + my * u;
                if ( y >= std::min( y0, y1 ) && y <= std::max( y0, y1 ) )
                    consider( u, c, seg, x0, y, d0 + std::fabs( y - y0 ) );
            }
            else if ( y0 == y1 && x0 != x1 && my != 0 )
            {
                REAL u = ( y0 - sy ) / my;
                REAL x = sx + mx * u;
                if ( x >= std::min( x0, x1 ) && x <= std::max( x0, x1 ) )
                    consider( u, c, seg, x, y0, d0 + std::fabs( x - x0 ) );
            }
        }
    }
    std::sort( found, found + nFound, []( Crossing const & a, Crossing const & b ) { return a.u < b.u; } );

    for ( int k = 0; k < nFound; ++k )
    {
        Crossing const & f = found[k];
        REAL tc = t0 + ( t1 - t0 ) * f.u;
        if ( f.owner < 0 )
        {
            // the rim kills
            me.diedWhileMoving = true;
            me.deathPosX = f.x;
            me.deathPosY = f.y;
            me.killer = -1;
            me.x = f.x;
            me.y = f.y;
            return;
        }
        if ( !EdgeDangerousFor( i, f.owner, f.seg, f.w, tc ) )
            continue;
        if ( f.owner != i )
        {
            Cycle & other = cycles[f.owner];
            REAL otherTime = WallTimeAt( f.owner, f.seg, f.w );
            if ( tc < otherTime * ( 1 - EPS ) )
            {
                // the other cycle got here later: we were first
                if ( f.w > other.distance * ( 1 + EPS ) )
                    continue;  // an extrapolated bit of wall
                if ( other.alive )
                {
                    bool saved = false;
                    if ( f.seg == NumWalls( f.owner ) - 1 )
                    {
                        // push the other cycle back to just before the crossing; its next frame
                        // simulates the collision again from its side
                        REAL d = other.DistanceSinceLastTurn() * .001f;
                        if ( d < .01f )
                            d = .01f;
                        REAL maxd = ( ( f.x - other.lastTurnX ) * other.dx + ( f.y - other.lastTurnY ) * other.dy ) * .5f;
                        if ( d > maxd )
                            d = maxd;
                        if ( d > 0 )
                        {
                            saved = true;
                            REAL sp = other.Speed() > .1f ? other.Speed() : .1f;
                            other.x = f.x - other.dx * d;
                            other.y = f.y - other.dy * d;
                            other.lastTime = otherTime - d / sp;
                            other.distance = f.w - d;
                            other.predX = other.x;
                            other.predY = other.y;
                            other.refreshSpace = true;
                        }
                    }
                    if ( !saved && me.verletSpeed >= 0 && s.wallsLength > 0 )
                    {
                        REAL dt = otherTime - tc;
                        REAL wallTimeLeft = s.wallsLength / me.verletSpeed - dt;
                        if ( wallTimeLeft < 0 )
                            continue;
                        REAL rubberToEat = wallTimeLeft * other.Speed();
                        other.rubber += rubberToEat;
                        if ( other.rubber > s.rubber )
                            other.rubber = s.rubber;
                        else
                            saved = true;
                    }
                    if ( !saved )
                    {
                        other.distance = f.w;
                        other.hunter = i;
                        other.hunterTime = tc;
                        Kill( f.owner, f.x, f.y );
                    }
                }
                continue;
            }
        }
        // sad but true
        me.diedWhileMoving = true;
        me.deathPosX = f.x;
        me.deathPosY = f.y;
        me.killer = f.owner;
        me.x = f.x;
        me.y = f.y;
        return;
    }
    me.x = nx;
    me.y = ny;
}

void World::Kill( int i, REAL px, REAL py )
{
    Cycle & c = cycles[i];
    if ( !c.alive )
        return;
    c.x = px;
    c.y = py;
    c.alive = false;
    c.deathTime = c.lastTime;
    c.deathX = px;
    c.deathY = py;
    c.expansion = 0;
    c.predX = px;
    c.predY = py;
    // the credit goes to the enemy whose wall influenced the cycle last
    int hunter = c.hunter;
    if ( hunter >= 0 && hunter != i && cycles[hunter].team != c.team )
        cycles[hunter].kills++;
}

bool World::TimestepCore( int i, REAL currentTime, bool calcAccel )
{
    Cycle & c = cycles[i];
    REAL lastSpeed = c.verletSpeed;
    REAL ts = currentTime - c.lastTime;

    if ( calcAccel )
        CalculateAcceleration( i );
    REAL lastAcceleration = c.acceleration;

    // the brake reservoir runs dry in this step: simulate up to that moment first
    if ( sg_recurseBrake && c.brakingReservoir > 0 && c.brakeUsage > 0 && c.brakingReservoir - ts * c.brakeUsage < 0 )
    {
        Guard guard( sg_recurseBrake );
        REAL brakeTime = c.lastTime + c.brakingReservoir / c.brakeUsage;
        if ( TimestepCore( i, brakeTime, false ) )
            return true;
        c.verletSpeed = c.Speed();
        c.lastTimestep = 0;
        c.brakingReservoir = -EPS;
        return TimestepCore( i, currentTime, true );
    }

    ApplyAcceleration( i, ts );
    ClampTo( ts, -10, 10 );
    REAL step = c.verletSpeed * ts;
    c.rubberSpeedFactor = 1;

    REAL rubberGranted = s.rubber, rubberEffectiveness = 1;
    rubberEffectiveness /= ( 1 + c.rubberMalus );

    if ( rubberGranted > c.rubber && s.rubberSpeed > 0 && step > -EPS && rubberEffectiveness > 0 )
    {
        REAL beta = ts * s.rubberSpeed;
        REAL neededSpace = 0, rubberFactor;
        if ( beta > .001f )
        {
            rubberFactor = 1 - std::exp( -beta );
            neededSpace = step / rubberFactor;
        }
        else
        {
            rubberFactor = beta;
            neededSpace = c.verletSpeed / s.rubberSpeed;
        }
        if ( rubberFactor > .999f )
            rubberFactor = .999f;
        if ( neededSpace < step * 3 || ts < -EPS )
            neededSpace = step * 3;

        REAL space = GetMaxSpaceAhead( i, neededSpace );
        if ( space < neededSpace )
        {
            REAL rubberStartSpace = c.verletSpeed / s.rubberSpeed;
            if ( space > rubberStartSpace && sg_recurseRubberStart )
            {
                // rubber is not active yet; simulate up to the moment it gets active
                Guard guard( sg_recurseRubberStart );
                REAL ratio = ( space - rubberStartSpace ) / step;
                if ( ratio > EPS && ratio < 1 - EPS )
                {
                    REAL t = c.lastTime + ( currentTime - c.lastTime ) * ratio;
                    c.verletSpeed = lastSpeed;
                    c.acceleration = lastAcceleration;
                    return TimestepCore( i, t, false ) || TimestepCore( i, currentTime, true );
                }
            }

            // the obstacle goes away during this step: simulate in two parts
            if ( c.spaceHit && c.spaceOwner >= 0 )
            {
                REAL tolerance = .001f;
                if ( !WallPointDangerous( c.spaceOwner, c.spaceWallDist, currentTime ) && currentTime > c.lastTime + tolerance )
                {
                    REAL distanceOffset = 0;
                    REAL speed = c.Speed();
                    if ( speed > 0 )
                        distanceOffset = space / speed;
                    REAL minTime = c.lastTime + distanceOffset;
                    REAL maxTime = currentTime + distanceOffset;
                    while ( minTime + tolerance < maxTime )
                    {
                        REAL midTime = .5f * ( minTime + maxTime );
                        if ( WallPointDangerous( c.spaceOwner, c.spaceWallDist, midTime ) )
                            minTime = midTime;
                        else
                            maxTime = midTime;
                    }
                    maxTime -= distanceOffset;
                    if ( sg_recurseObstacle )
                    {
                        Guard guard( sg_recurseObstacle );
                        c.verletSpeed = lastSpeed;
                        c.acceleration = lastAcceleration;
                        return TimestepCore( i, maxTime, false ) || TimestepCore( i, currentTime, true );
                    }
                }
            }

            REAL rubberStep = space * rubberFactor;
            if ( rubberStep > step )
                rubberStep = step;
            if ( step < 0 )
                step = 0;
            REAL rubberneeded = step - rubberStep;
            if ( rubberneeded < 0 )
                rubberneeded = 0;

            REAL rubberAvailable = ( rubberGranted - c.rubber ) * rubberEffectiveness;
            if ( rubberneeded > rubberAvailable )
            {
                // rubber runs out in this step: simulate up to that moment first
                REAL ratio = rubberAvailable / rubberneeded;
                if ( ratio > .01f && ratio < .99f && currentTime - c.lastTime > .001f && sg_recurseRunOut )
                {
                    REAL runOutTime = c.lastTime + ( currentTime - c.lastTime ) * ratio;
                    Guard guard( sg_recurseRunOut );
                    c.verletSpeed = lastSpeed;
                    c.acceleration = lastAcceleration;
                    if ( TimestepCore( i, runOutTime, false ) )
                        return true;
                    return TimestepCore( i, currentTime, true );
                }
                rubberneeded = rubberAvailable;
            }

            c.rubber += rubberneeded / rubberEffectiveness;
            if ( step > 0 )
                c.rubberSpeedFactor = 1 - rubberneeded / step;
            else
                c.rubberSpeedFactor = space / neededSpace;
            if ( c.rubberSpeedFactor < 0 )
                c.rubberSpeedFactor = 0;
            step -= rubberneeded;
            if ( step < 0 )
                step = 0;
        }
    }

    if ( step < 0 )
    {
        REAL mn = -c.DistanceSinceLastTurn();
        if ( step < mn )
            step = mn;
    }

    REAL lastX = c.x, lastY = c.y;
    REAL nx = c.x, ny = c.y;
    if ( c.verletSpeed > 0 )
    {
        nx = c.x + c.dx * step;
        ny = c.y + c.dy * step;
    }
    c.diedWhileMoving = false;
    Move( i, nx, ny, c.lastTime, currentTime );
    Cycle & cc = cycles[i];  // (Move may have touched other cycles, never this vector's layout)
    cc.distance += step;

    if ( cc.diedWhileMoving )
    {
        cc.rubberSpeedFactor = 0;
        if ( rubberEffectiveness <= 0 || step >= ( rubberGranted - cc.rubber ) * rubberEffectiveness )
        {
            // no straw left
            cc.rubber = rubberGranted;
            cc.distance += ( cc.x - lastX ) * cc.dx + ( cc.y - lastY ) * cc.dy;
            cc.lastTime = currentTime;
            return true;
        }
        // rubber saves the cycle this time: it stays where it was
        cc.diedWhileMoving = false;
        cc.x = lastX;
        cc.y = lastY;
        cc.rubber += step / rubberEffectiveness;
        if ( cc.rubber < 0 )
            cc.rubber = 0;
    }

    if ( rubberEffectiveness > 0 )
        cc.rubber += cc.rubberUsage * ts * cc.verletSpeed / rubberEffectiveness;
    cc.rubberUsage = 0;

    if ( cc.rubber > rubberGranted )
    {
        cc.diedWhileMoving = true;
        cc.deathPosX = cc.x;
        cc.deathPosY = cc.y;
        cc.killer = -1;
        cc.lastTime = currentTime;
        return true;
    }

    cc.brakingReservoir -= cc.brakeUsage * ts;
    ClampTo( cc.brakingReservoir, 0, 1 );

    if ( s.rubberTime > 0 )
        cc.rubber /= ( 1 + ts / s.rubberTime );
    else
        cc.rubber = 0;
    cc.rubberMalus = 0;
    if ( cc.rubber >= rubberGranted )
        cc.rubber = rubberGranted;

    cc.lastTime = currentTime;
    return false;
}

void World::Timestep( int i, REAL currentTime )
{
    Cycle & c = cycles[i];
    if ( !c.alive )
        return;

    // PreparePredictPosition: how far ahead the wall will be drawn after this step
    REAL rubberStart = c.verletSpeed / s.rubberSpeed;
    REAL maxSpaceReport = s.predictAhead * c.verletSpeed + rubberStart;
    c.maxSpaceMaxCast = maxSpaceReport;

    c.refreshSpace = true;
    ClampTo( c.rubber, 0, s.rubber );

    if ( currentTime > c.lastTime )
    {
        TimestepCore( i, currentTime, true );
        if ( cycles[i].diedWhileMoving )
        {
            Cycle & d = cycles[i];
            d.diedWhileMoving = false;
            if ( d.killer >= 0 && d.killer != i && cycles[d.killer].team != d.team )
            {
                d.hunter = d.killer;
                d.hunterTime = currentTime;
            }
            Kill( i, d.deathPosX, d.deathPosY );
            return;
        }
    }

    // CalculatePredictPosition
    Cycle & cc = cycles[i];
    cc.predX = cc.x;
    cc.predY = cc.y;
    REAL spaceAhead = GetMaxSpaceAhead( i, maxSpaceReport ) - rubberStart;
    if ( spaceAhead > 0 )
    {
        cc.predX = cc.x + cc.dx * spaceAhead;
        cc.predY = cc.y + cc.dy * spaceAhead;
    }
}

void World::Explode( int i )
{
    Cycle & c = cycles[i];
    int const steps = 5;
    REAL const expansionTime = .2f;
    int current = int( REAL( steps ) * ( ( time - c.deathTime ) / expansionTime ) ) + 1;
    if ( current > steps )
        current = steps;
    if ( current <= c.expansion )
        return;
    c.expansion = current;
    REAL radius = s.explosionRadius * std::sqrt( REAL( current ) / steps );
    if ( radius <= 0 || time >= c.deathTime + 4 )
        return;
    REAL ex = c.deathX, ey = c.deathY;
    for ( int o = 0; o < int( cycles.size() ); ++o )
    {
        Cycle & cy = cycles[o];
        int n = NumWalls( o );
        for ( int seg = 0; seg < n; ++seg )
        {
            REAL x0, y0, d0, x1, y1, d1;
            WallEnds( o, seg, x0, y0, d0, x1, y1, d1, false );
            REAL len = std::fabs( x1 - x0 ) + std::fabs( y1 - y0 );
            if ( len <= 0 )
                continue;
            // perpendicular distance to the wall's line, and where along it the closest point is
            REAL perp, along;
            if ( x0 == x1 )
            {
                perp = ex - x0;
                along = ( ey - y0 ) * ( y1 > y0 ? 1 : -1 );
            }
            else
            {
                perp = ey - y0;
                along = ( ex - x0 ) * ( x1 > x0 ? 1 : -1 );
            }
            REAL r2 = radius * radius - perp * perp;
            if ( r2 < 0 )
                continue;
            REAL h = std::sqrt( r2 );
            REAL start = d0 + along - h, end = d0 + along + h;
            // only what was built before the explosion: never past the owner's current distance
            REAL limit = cy.alive ? cy.distance : d1;
            if ( end > d1 )
                end = d1;
            if ( end > limit )
                end = limit;
            if ( start < d0 )
                start = d0;
            if ( end > start )
                cy.holes.push_back( std::make_pair( start, end ) );
        }
    }
}

void World::Frame()
{
    time += s.frameDt;
    // explosions were added to the object list after the cycles, so they move first
    for ( int i = 0; i < int( cycles.size() ); ++i )
        if ( !cycles[i].alive && cycles[i].expansion < 5 && time > cycles[i].deathTime )
            Explode( i );
    // the engine simulates its object list from the back
    for ( int i = int( cycles.size() ) - 1; i >= 0; --i )
        Timestep( i, time );
}

void World::Step( int const * actions )
{
    // gCycle::Act drops input while the game timer is not running yet, which it is not at time 0
    if ( time > 1E-6f )
        for ( int i = 0; i < int( cycles.size() ); ++i )
            if ( cycles[i].alive )
                Act( i, actions[i] );
    // frames up to the next decision time
    REAL next = time + s.decision;
    while ( time + 1E-5f < next )
        Frame();
}

int World::Alive() const
{
    int n = 0;
    for ( size_t i = 0; i < cycles.size(); ++i )
        n += cycles[i].alive ? 1 : 0;
    return n;
}

int World::TeamsAlive() const
{
    int teams[64], n = 0;
    for ( size_t i = 0; i < cycles.size(); ++i )
    {
        if ( !cycles[i].alive )
            continue;
        bool seen = false;
        for ( int k = 0; k < n; ++k )
            seen = seen || teams[k] == cycles[i].team;
        if ( !seen && n < 64 )
            teams[n++] = cycles[i].team;
    }
    return n;
}

bool World::Over() const
{
    return roundTotal > 1 ? TeamsAlive() <= 1 : Alive() == 0;
}
}
