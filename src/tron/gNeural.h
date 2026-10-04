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

#ifndef ArmageTron_gNEURAL_H
#define ArmageTron_gNEURAL_H

#include "defs.h"

class gAIPlayer;

//! Bridge that lets an external process (a neural network) drive some of the AI players.
//! Configured with NEURAL_SOCKET (unix socket path) and NEURAL_SLOTS (how many AI players
//! it controls); see the Armagetron AI project for the wire protocol.
namespace gNeural
{
    //! true if neural control is configured
    bool Active();

    //! true if the given AI player is driven by the external brain
    bool Controls( gAIPlayer const * player );

    //! called after every world timestep on the server; makes decisions at fixed intervals
    void Timestep( REAL time );

    //! called after the cycles of a new round have been spawned
    void NewRound();

    //! the lockstep frame time; 0 means the game runs in real time
    REAL LockstepDT();
}

#endif
