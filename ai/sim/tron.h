// A fast, cloneable Armagetron arena for the teacher's search.
//
// The physics is a port of the engine's own (src/tron/gCycleMovement.cpp, gCycle.cpp, gWall.cpp,
// gExplosion.cpp) for the settings the AI plays with: a dedicated server, no lag, the default
// cycle physics, finite trails. The observation is a port of src/tron/gNeural.cpp's, byte for
// byte the same layout, so the network cannot tell the simulator from the game.
//
// Floats, like the engine (REAL is float there), so rounding drifts the same way.

#ifndef TRON_SIM_H
#define TRON_SIM_H

#include <cstdint>
#include <utility>
#include <vector>

namespace tron
{
typedef float REAL;
const REAL EPS = REAL( 1E-7 );

enum { S_NONE = 0, S_RIM = 1, S_ENEMY = 2, S_TEAMMATE = 3, S_SELF = 4 };  // gSensorWallType
enum { ACT_STRAIGHT = 0, ACT_LEFT = 1, ACT_RIGHT = 2, ACT_BRAKE = 3 };

struct Settings
{
    // movement (settings.cfg defaults; CYCLE_DELAY_BONUS is .95 on a dedicated server)
    REAL speed = 20, speedMin = .25f, speedMax = 0, decayBelow = 5, decayAbove = .1f, startSpeed = 20;
    REAL accel = 15, accelOffset = 2, wallNear = 6;
    REAL accelSelf = 1, accelTeam = 1, accelEnemy = 1, accelRim = 0, accelSlingshot = 1, accelTunnel = 1;
    REAL brake = 30, brakeRefill = .1f, brakeDeplete = 1;
    REAL delay = .1f, delayBonus = .95f, turnSpeedFactor = .95f;
    // rubber
    REAL rubber = 1, rubberSpeed = 40, rubberTime = 10;
    REAL rubberMinDistance = .001f, rubberMinDistanceRatio = .0001f, rubberMinDistanceReservoir = .005f;
    REAL rubberMinDistanceUnprepared = .005f, rubberMinDistancePreparation = .2f, rubberMinAdjust = .05f;
    // trails (SP_ settings the training arenas use)
    REAL wallsLength = 600, wallsStayUp = 8, explosionRadius = 4;
    // time: physics frames of LOCKSTEP_DT, decisions every NEURAL_DECISION_INTERVAL
    REAL frameDt = .025f, decision = .05f;
    // dedicated server wall prediction: MAX_SIMULATE_AHEAD + 2 x lazy lag (DEDICATED_FPS 40, idle factor 2)
    REAL predictAhead = .175f;
    // observation
    REAL localCell = 2;
};

struct Coord
{
    REAL x, y, dist, time;
};

//! what a sensor ray found
struct Hit
{
    REAL t = 0;        //!< ray parameter: the hit point is start + dir * t
    int type = S_NONE;
    int owner = -1;    //!< cycle whose wall it is; -1 for the rim or nothing
    int seg = -1;      //!< which wall of the owner (index into its trail)
    REAL wallLen = 0;  //!< length of the wall that was hit
    REAL x = 0, y = 0; //!< the hit point
    REAL wallDist = 0; //!< trail distance at the hit point
};

struct Cycle
{
    int team = 0;
    bool alive = true;
    REAL deathTime = 0, deathX = 0, deathY = 0;
    int expansion = 0;  //!< explosion steps done after death (5 = finished)
    REAL x = 0, y = 0, dx = 0, dy = 1;
    REAL verletSpeed = 20, acceleration = 0, lastTimestep = 0;
    bool braking = false;
    REAL brakingReservoir = 1, brakeUsage = 0;
    REAL rubber = 0, rubberUsage = 0, rubberSpeedFactor = 1, rubberMalus = 0;
    REAL lastTurnLeft = -10, lastTurnRight = -10;
    REAL lastTime = 0, distance = 0;
    REAL lastTurnX = 0, lastTurnY = 0;
    REAL predX = 0, predY = 0;     //!< where the current wall is drawn up to (dedicated-server wall prediction)
    REAL maxSpaceMaxCast = 0;      //!< extra raycast length requested for the prediction
    std::vector< Coord > trail;    //!< turn points, oldest first; the current wall runs from trail.back()
    std::vector< std::pair< REAL, REAL > > holes;  //!< trail-distance intervals blown away
    int hunter = -1;               //!< the enemy whose wall last influenced this cycle
    REAL hunterTime = -1000;
    int kills = 0;
    // GetMaxSpaceAhead cache
    bool refreshSpace = true;
    bool spaceHit = false;
    REAL spaceX = 0, spaceY = 0, spaceOffset = 0;
    int spaceOwner = -1, spaceSeg = -1;
    REAL spaceWallDist = 0;
    // set by a move that crossed a dangerous wall
    bool diedWhileMoving = false;
    REAL deathPosX = 0, deathPosY = 0;
    int killer = -1;

    REAL Speed() const
    {
        REAL r = verletSpeed + .5f * lastTimestep * acceleration;
        return r > 0 ? r : 0;
    }
    REAL LastTurnTime() const { return lastTurnRight > lastTurnLeft ? lastTurnRight : lastTurnLeft; }
    REAL DistanceSinceLastTurn() const { return ( x - lastTurnX ) * dx + ( y - lastTurnY ) * dy; }
};

//! one round in one arena
class World
{
public:
    Settings s;
    REAL minX = 0, minY = 0, maxX = 500, maxY = 500;
    REAL time = 0;
    int roundTotal = 0;
    std::vector< Cycle > cycles;

    //! a fresh round: SIZE_FACTOR arena, N cycles on the map's spawn points (square-1.0.1),
    //! spawn k given to cycle (k + rotate) % n; TEAMS[i] (or each its own team when null)
    void Reset( REAL sizeFactor, int n, int rotate = 0, int const * teams = 0 );
    //! place cycle I explicitly (for checks against recordings)
    void Place( int i, REAL x, REAL y, REAL dx, REAL dy );

    //! the moves allowed now (gNeural.cpp's ActionMask)
    unsigned ActionMask( int i ) const;
    //! apply a move at a decision time (gNeural.cpp's Apply)
    void Act( int i, int action );
    //! advance every cycle by one physics frame
    void Frame();
    //! one decision interval: ACTIONS (one per cycle, ignored for the dead) then frames up to the next decision
    void Step( int const * actions );

    int Alive() const;
    int TeamsAlive() const;
    bool Over() const;  //!< gNeural.cpp's round-over rule

    //! a sensor ray from OWNER's point of view (-1: nobody's, sees every wall)
    Hit Cast( int owner, REAL px, REAL py, REAL ddx, REAL ddy, REAL range, REAL inverseSpeed = 0 ) const;

    //! the observation gNeural.cpp builds for cycle I: four 64x64 maps and the scalars
    void Observe( int i, uint8_t * local, uint8_t * close, uint8_t * global, uint8_t * territory, float * scalars ) const;

    static int const kGrid = 64, kScalars = 96;

    //! is the point at trail distance W of cycle C's walls solid at time T (no owner exclusions)
    bool WallPointDangerous( int c, REAL w, REAL t ) const;

private:
    void Timestep( int i, REAL currentTime );
    bool TimestepCore( int i, REAL currentTime, bool calcAccel );
    void CalculateAcceleration( int i );
    void ApplyAcceleration( int i, REAL dt );
    REAL GetMaxSpaceAhead( int i, REAL maxReport );
    bool DoTurn( int i, int dir );
    void Move( int i, REAL nx, REAL ny, REAL t0, REAL t1 );
    void Kill( int i, REAL px, REAL py );
    void Explode( int i );
    bool EdgeDangerousFor( int self, int owner, int seg, REAL w, REAL t ) const;
    int NumWalls( int c ) const;
    void WallEnds( int c, int seg, REAL & x0, REAL & y0, REAL & d0, REAL & x1, REAL & y1, REAL & d1, bool predicted ) const;
    REAL WallTimeAt( int c, int seg, REAL w ) const;
    REAL TurnDelay( int i ) const;
    REAL NextTurn( int i ) const;
};
}

#endif
