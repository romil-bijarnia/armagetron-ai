// The teacher: many self-play rounds at once in the simulator, every player's moves improved by a
// search that looks ahead with the network, and every decision of a learning player written to a
// replay ring the trainer reads from.
//
// The network itself lives in Python. Collect() runs every round until it needs network outputs and
// hands over a batch of observations; Feed() takes the outputs back. One batch mixes the rounds'
// real decisions and their searches' look-ahead positions.
//
// Search (per searched decision, joint over all players): a tree whose edges are MACRO decisions
// (0.1 s by default, the turn delay). Each player keeps its own statistics at every node
// (decoupled). At the root each player picks its candidates with Gumbel noise and narrows them by
// sequential halving; below the root each player follows Gumbel MuZero's deterministic rule. The
// training target is softmax(logits + sigma(completed Q)), which improves on the network's policy
// even with few simulations (Danihelka et al., 2022).

#ifndef TRON_SELFPLAY_H
#define TRON_SELFPLAY_H

#include "tron.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace tron
{
int const kMaxPlayers = 8;
int const kMapBytes = 4 * World::kGrid * World::kGrid;  // one decision's four maps
int const kActions = 4;

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
    int threads = 8;
};

//! where a learning agent's samples go (shared memory owned by the trainer)
struct Ring
{
    uint8_t * maps = 0;     //!< [cap][4][64][64]
    float * scalars = 0;    //!< [cap][96]
    uint8_t * mask = 0;     //!< [cap]
    float * policy = 0;     //!< [cap][4]
    uint8_t * hasPolicy = 0;
    float * value = 0;
    float * aux = 0;        //!< [cap][2]: territory share in auxHorizon decisions, dead within auxHorizon
    int64_t * prev = 0;     //!< global index of the same player's previous decision, -1 at a spawn
    int64_t * gen = 0;      //!< global index of the sample a slot holds
    uint8_t * ready = 0;    //!< targets filled in (the round is over)
    int64_t * counter = 0;  //!< samples ever written
    int64_t cap = 0;
    bool Valid() const { return cap > 0 && counter; }
};

struct Node
{
    World w;
    int depth = 0;
    bool evaluated = false;
    bool done[kMaxPlayers];    //!< the player's round is over here (dead, or the round ended)
    float util[kMaxPlayers];   //!< ... and what it got
    float logits[kMaxPlayers][kActions];
    float value[kMaxPlayers];
    unsigned mask[kMaxPlayers];
    int N[kMaxPlayers][kActions];
    float W[kMaxPlayers][kActions];
    std::vector< std::pair< uint32_t, int > > kids;  //!< joint action code -> node
};

struct Request
{
    int node;    //!< -1: the real decision
    int player;
};

class Game
{
public:
    // setup
    int n = 0;
    float size = -3;
    int agents[kMaxPlayers];
    bool learn[kMaxPlayers];
    uint64_t rng = 1;
    // state
    World world;
    bool active = false, finished = false;
    int decision = 0;
    int endDecision[kMaxPlayers];
    float outcome[kMaxPlayers];
    std::vector< uint8_t > prevMaps, curMaps;  //!< per player
    std::vector< float > curScalars;
    bool havePrev[kMaxPlayers];
    // network outputs at the real decision
    float rootLogits[kMaxPlayers][kActions];
    float rootValue[kMaxPlayers];
    // search
    bool searching = false;
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
    std::vector< uint8_t > reqMaps;     //!< [k][8][64][64]: current maps then the previous decision's
    std::vector< float > reqScalars;    //!< [k][96]
    std::vector< uint8_t > reqMask;
    // the trajectory of each learning player: (global ring index, decision, territory share)
    struct Entry { int64_t index; int decision; float share; };
    std::vector< Entry > traj[kMaxPlayers];
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
    int Collect( uint8_t * maps, float * scalars, uint8_t * masks, int * agentIds, int * gameIds, int maxBatch );
    void Feed( float const * logits, float const * values, int batch );

    // stats since the last call
    std::atomic< int64_t > statDecisions{ 0 }, statSearches{ 0 }, statSims{ 0 }, statSamples{ 0 };

private:
    std::vector< int > batchGames_;  //!< game of each request in the last batch, in order
    void Advance( Game & g );
    void BeginDecision( Game & g );
    void FinishDecision( Game & g, int const * acts );
    void BeginSearch( Game & g );
    void RunSimulations( Game & g );
    void Backup( Game & g, int leaf );
    void Expand( Game & g, int parent, uint32_t code, int & child );
    void Ask( Game & g, int node, int player, World const & now, World const & before );
    void Record( Game & g, int p, bool searched, float const * policy );
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
