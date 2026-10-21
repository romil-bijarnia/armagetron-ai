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

#ifndef ArmageTron_gFIGHTER_H
#define ArmageTron_gFIGHTER_H

#include "defs.h"

class gAIPlayer;

//! The space fighter as the mind of the stock bots (AI_FIGHTER): every decision interval it
//! copies the arena into the AI project's simulator (ai/sim), looks a few turns ahead over the
//! real physics on a worker thread and plays the move that holds the most territory.
namespace gFighter
{
    //! true if this AI player is driven by the fighter
    bool Controls( gAIPlayer const * player );

    //! called instead of the stock AI's thinking
    void Timestep( gAIPlayer * player, REAL time );

    //! called after the cycles of a new round have been spawned
    void NewRound();

    //! stop and join the worker threads
    void Shutdown();
}

#endif
