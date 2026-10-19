// The teacher: many self-play rounds at once in the simulator, every player's moves improved by a
// search that looks ahead with the network, and every decision of a learning player written to a
// replay ring the trainer reads from.
//
// The network itself lives in Python. Collect() runs every round until it needs network outputs and
// hands over a batch of observations; Feed() takes the outputs back. One batch mixes the rounds'
// real decisions and their searches' look-ahead positions.
//
// Search (per searched decision, joint over network players): a tree whose edges are at most MACRO
// decisions (0.1 s by default, the turn delay). A deterministic hunter's cached control is exact
// until its next replan; hunter games stop the tree there and bootstrap from the network instead
// of inventing replies or nesting the expensive hunter planner inside every simulation. Edges
// retain their actual duration for discounting and one-decision observation history.
// Each network player keeps its own statistics at every node
// (decoupled). At the root each player picks its candidates with Gumbel noise and narrows them by
// sequential halving; below the root each player follows Gumbel MuZero's deterministic rule. The
// training target is softmax(logits + sigma(completed Q)), which improves on the network's policy
// even with few simulations (Danihelka et al., 2022).

#ifndef TRON_SELFPLAY_H
#define TRON_SELFPLAY_H

#include "tron.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <vector>

namespace tron
{
int const kMaxPlayers = 8;
int const kMapBytes = 4 * World::kGrid * World::kGrid;  // one decision's four maps
int const kActions = 6;
int const kAux = 3;
int const kMemory = 256;
int const kContext = kActions + 1; //!< previous executed control and elapsed decision intervals
int const kAgentHunter = -1;  //!< a seat played by the built-in space fighter; it asks no network

// Telemetry describes the last move actually sent to World::Step. It never selects a move
// or consumes randomness; a search currently in progress is separate from that last move.
enum { WATCH_NONE = 0, WATCH_SEARCH = 1, WATCH_SAMPLED = 2, WATCH_GREEDY = 3, WATCH_HUNTER = 4 };
struct DecisionWatch
{
    int action = -1, decision = -1, selection = WATCH_NONE;
    unsigned mask = 0;
    int64_t policyUpdate = -1;
    float policy[kActions] = {}, search[kActions] = {}, value = 0;
    bool hunterReplanned = false;
    float observedAt = 0;
    std::array< float, kContext > context{};
    // Exact inputs for the last executed root decision, never a hypothetical search node.
    std::vector< uint8_t > maps, previous;
    std::vector< float > scalars;
};

struct SPConfig
{
    int sims = 32;              //!< simulations per searched decision
    int macro = 2;              //!< decisions per tree edge
    float gamma = .999f;        //!< discount per decision
    float cVisit = 50, cScale = .1f;  //!< Gumbel sigma
    float searchProb = .25f;    //!< chance that a decision is searched
    float temperature = 1;      //!< sampling temperature of unsearched decisions (0: best move)
    float gumbel = 1;           //!< scale of the root's Gumbel noise (0: none)
    int maxDecisions = 6000;    //!< a round is a draw after this many decisions (300 s)
    int auxHorizon = 40;        //!< decisions ahead for the auxiliary targets (2 s)
    int valueSteps = 60;        //!< a sample's value target is the discounted value estimate this many
                                //!< decisions later (3 s; at least auxHorizon + 1), or the outcome if the
                                //!< round ends sooner: it is ready long before the shared ring overwrites it
    int threads = 8;
    int hunterDepth = 2;        //!< plies the space fighter looks ahead over the real physics (1 or 2)
    int hunterReplan = 2;       //!< decisions between the space fighter's decisions (it holds in between)
    int hunterPly = 4;          //!< decisions per ply of its look-ahead (0.2 s: a turn and the drive after it)
    int hunterProbe = 6;        //!< decisions it drives straight on past the last ply, to feel a wall coming
};

//! where a learning agent's samples go (shared memory owned by the trainer)
struct Ring
{
    uint8_t * maps = 0;     //!< [cap][4][64][64]
    float * scalars = 0;    //!< [cap][108]
    uint8_t * mask = 0;     //!< [cap]
    float * policy = 0;     //!< [cap][kActions]
    uint8_t * hasPolicy = 0;
    float * value = 0;
    float * aux = 0;        //!< [cap][kAux]: territory share, dead within auxHorizon, credited kill and survive that horizon
    int64_t * prev = 0;     //!< global index of the same player's previous decision, -1 at a spawn
    int64_t * gen = 0;      //!< global index of the sample a slot holds
    uint8_t * ready = 0;    //!< targets filled in (the round is over)
    int64_t * counter = 0;  //!< samples ever written
    int64_t * future = 0;   //!< [cap]: ring index of the sample whose territory map is the ownership target
    uint8_t * heading = 0;  //!< [cap]: the player's heading quadrant (0 up, 1 right, 2 down, 3 left)
    int64_t * generation = 0; //!< live exploiter generation; never a cached recurrent state
    float * context = 0;   //!< [cap][kContext], retained independently of target readiness
    int64_t cap = 0;
    bool Valid() const { return cap > 0 && counter; }
};

struct Node
{
    World w;
    int depth = 0, elapsed = 0;  //!< edges and actual decisions from the root
    bool evaluated = false, cutoff = false;  //!< cutoff leaves bootstrap at the hunter's next replan
    bool done[kMaxPlayers];    //!< the player's round is over here (dead, or the round ended)
    float util[kMaxPlayers];   //!< ... and what it got
    float logits[kMaxPlayers][kActions];
    float value[kMaxPlayers];
    unsigned mask[kMaxPlayers];
    std::array< float, kMemory > memory[kMaxPlayers]{}; //!< after this node's observation
    std::array< float, kContext > context[kMaxPlayers]{};
    // Each branch carries the known hunter control; it never simulates an unknown replan.
    int hunterAct[kMaxPlayers], hunterAt[kMaxPlayers];
    int N[kMaxPlayers][kActions];
    float W[kMaxPlayers][kActions];
    std::vector< std::pair< uint32_t, int > > kids;  //!< joint action code -> node
};

struct Request
{
    int node;    //!< -1: the real decision
    int player;
    std::array< float, kMemory > memory{}; //!< before the first requested observation
    std::vector< uint8_t > maps, masks; //!< consecutive observations along this edge
    std::vector< float > scalars, contexts;
};

class Game
{
public:
    // setup
    int n = 0;
    float size = -3;
    int agents[kMaxPlayers];
    bool learn[kMaxPlayers];
    int64_t generation[kMaxPlayers]{};
    std::array< float, kMemory > memory[kMaxPlayers]{};
    std::array< float, kContext > context[kMaxPlayers]{};
    uint64_t rng = 1;
    // state
    World world;
    bool active = false, finished = false;
    int decision = 0;
    int endDecision[kMaxPlayers];
    float outcome[kMaxPlayers];
    int hunterAct[kMaxPlayers];   //!< the space fighter's last planned move ...
    int hunterAt[kMaxPlayers];    //!< ... and the decision it was planned at
    std::vector< uint8_t > prevMaps, curMaps;  //!< per player
    std::vector< float > curScalars;
    bool havePrev[kMaxPlayers];
    // network outputs at the real decision
    float rootLogits[kMaxPlayers][kActions];
    float rootValue[kMaxPlayers];
    int64_t rootUpdate[kMaxPlayers];  //!< checkpoint used for the real decision's network evaluation
    DecisionWatch lastChoice[kMaxPlayers];
    int64_t sourceCounts[5] = {};  //!< executed player moves in this round, indexed by WATCH_*
    // search
    bool searching = false;
    int searchUntil = -1;  //!< absolute next hunter replan; -1 for unrestricted network-only search
    std::vector< Node > nodes;
    int simsDone = 0;
    std::vector< std::pair< int, uint32_t > > path;  //!< (node, joint action) from the root
    int pendingNode = -1;
    float gumbel[kMaxPlayers][kActions];
    std::vector< int > cands[kMaxPlayers];
    int phase[kMaxPlayers], phases[kMaxPlayers], perCand[kMaxPlayers];
    int phaseVisits[kMaxPlayers][kActions];
    // driving: 0 = start a decision, 1 = waiting for the decision's network outputs,
    // 2 = searching, 3 = waiting for a look-ahead position's outputs
    int stage = 0;
    bool waiting = false, fed = false;
    bool died[kMaxPlayers];
    // what Collect handed out and Feed must answer
    std::vector< Request > requests;
    std::vector< float > outLogits, outValue;
    std::vector< float > outMemory;
    std::vector< int64_t > outUpdates;
    std::vector< uint8_t > reqMaps;     //!< [k][8][64][64]: current maps then the previous decision's
    std::vector< float > reqScalars;    //!< [k][108]
    std::vector< uint8_t > reqMask;
    // the trajectory of each learning player: (global ring index, decision, territory share)
    struct Entry { int64_t index; int decision; float share, value; int kills; };
    std::vector< Entry > traj[kMaxPlayers];
    size_t filled[kMaxPlayers];  //!< entries of traj whose targets are written
    // stats
    int searches = 0, sims = 0;
};

class SelfPlay
{
public:
    SPConfig cfg;
    std::vector< Game > games;
    Ring rings[4];  //!< one per learning agent id (0..3)

    explicit SelfPlay( int nGames );
    void Start( int g, float size, int n, int const * agents, int const * learn, int rotate, uint64_t seed );
    //! advance every round to its next network evaluation; returns the batch size
    //! ROOTS (may be null) gets 1 for a request that is a real decision (not a look-ahead position)
    int Collect( uint8_t * maps, float * scalars, uint8_t * masks, int * agentIds, int * gameIds, int maxBatch,
                 uint8_t * roots = 0, float * memory = 0, int * lengths = 0,
                 uint8_t * sequenceMaps = 0, float * sequenceScalars = 0, uint8_t * sequenceMasks = 0,
                 float * sequenceContexts = 0 );
    //! LOGITS has six entries per request; the world's action mask determines legal moves.
    //! VALUES guide search and form the n-step targets of real decisions.
    void Feed( float const * logits, float const * values, int batch, int64_t const * updates = 0,
               float const * memory = 0 );

    // stats since the last call
    std::atomic< int64_t > statDecisions{ 0 }, statSearches{ 0 }, statSims{ 0 }, statSamples{ 0 };

private:
    std::vector< int > batchGames_;  //!< game of each request in the last batch, in order
    void Advance( Game & g );
    void BeginDecision( Game & g );
    void Decide( Game & g );
    void WatchDecision( Game & g, int p, int action, int selection, float const * search = 0 );
    int HunterAction( Game & g, int p );
    int HunterAction( World const & w, int decision, int p, int & action, int & plannedAt ) const;
    int HunterMove( World const & w, int p ) const;
    float HunterEval( World const & w, int p, int e ) const;
    void FinishDecision( Game & g, int const * acts );
    void BeginSearch( Game & g );
    void RunSimulations( Game & g );
    void Backup( Game & g, int leaf );
    void Expand( Game & g, int parent, uint32_t code, int & child );
    void Ask( Game & g, int node, int player, World const & now, World const & before );
    void Record( Game & g, int p, bool searched, float const * policy );
    void Fill( Game & g, int p, Ring & R );
    void FinishPlayer( Game & g, int p );
    void Finish( Game & g );
    void MarkDeaths( Game & g, World const & before, World const & after, int decision );
    float RootScore( Game & g, int p, int a ) const;
    void ImprovedPolicy( Game const & g, Node const & nd, int p, float * out ) const;
    int SelectRoot( Game & g, int p );
    int SelectInner( Game const & g, Node const & nd, int p ) const;
    uint32_t Code( int const * actions, int n ) const;
};
}

#endif
