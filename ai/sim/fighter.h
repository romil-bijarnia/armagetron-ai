// The space fighter: a built-in opponent that asks no network. A shallow minimax over the
// real physics, scored by the territory share (the cells it reaches before the nearest enemy).
// It cuts, seals and takes space because that is what the score rewards; nothing else is
// scripted. The teacher uses it as a sparring partner (selfplay.cpp) and the game as the mind
// of its stock bots (src/tron/gFighter.cpp).
#ifndef TRON_FIGHTER_H
#define TRON_FIGHTER_H

#include "tron.h"
#include <cstdint>

namespace tron
{
int const kMaxPlayers = 8;
int const kActions = 6;

struct FighterConfig
{
    int depth = 2;  //!< plies it looks ahead over the real physics (1 or 2)
    int ply = 4;    //!< decisions per ply (0.2 s: a turn and the drive after it)
    int probe = 6;  //!< decisions it drives straight on past the last ply, to feel a wall coming
};

//! inside a tree edge a turn is followed by driving straight; braking keeps braking
inline int Hold( int a )
{
    return ( a == ACT_BRAKE || a == ACT_LEFT_BRAKE || a == ACT_RIGHT_BRAKE ) ? ACT_BRAKE : ACT_STRAIGHT;
}

//! the share of the arena cycle P reaches first, from its territory map
float TerritoryShare( uint8_t const * territory );

//! P plays A, E answers EA, everyone else drives on, for MACRO decisions
void FighterPly( World & w, int n, int p, int a, int e, int ea, int macro );

//! how good the position is for P against E (-1 dead, +1 E dead, else territory and the probe)
float FighterEval( World const & w, int p, int e, FighterConfig const & cfg );

//! the move for P now
int FighterMove( World const & w, int p, FighterConfig const & cfg );
}

#endif
