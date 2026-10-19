#ifndef ARMAGETRON_CYCLE_INPUT_H
#define ARMAGETRON_CYCLE_INPUT_H

#include <cstddef>

namespace gCycleInput
{
// Local input policy, independent of the simulation's turn-memory rules. A full
// buffer consumes another press without changing any already ordered turn;
// that turn may belong to chat auto-drive rather than the human input path.
inline bool BufferFull(int mode, std::size_t pendingTurns)
{
    return mode > 0 && pendingTurns != 0;
}
}

#endif
