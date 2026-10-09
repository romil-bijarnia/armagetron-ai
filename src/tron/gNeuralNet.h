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

#ifndef ArmageTron_gNEURALNET_H
#define ArmageTron_gNEURALNET_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

//! The trained policy network, run inside the game: no Python, no socket.
//!
//! Loads the weights written by `arma-export` (ai/arma_ai/export.py) and turns one
//! observation (the same local map, arena map and exact numbers the neural bridge sends to the
//! trainer) into one of the four moves. Plain C++ on purpose, so it has no dependency on the
//! rest of the engine and can be checked against PyTorch on its own.
namespace gNeuralNet
{
    struct Conv
    {
        int cout, cin, k, stride, pad;
        std::vector< float > w; //!< [cout][cin][k][k]
        std::vector< float > b;
    };

    struct Linear
    {
        int out, in;
        std::vector< float > w; //!< [out][in]
        std::vector< float > b;
    };

    class Net
    {
    public:
        Net();

        //! reads a policy file; on failure returns false and says why in ERROR
        bool Load( std::istream & in, std::string & error );
        bool Loaded() const { return loaded_; }
        int Update() const { return update_; } //!< the training update the weights come from

        //! one decision. LOCAL and GLOBAL are grid*grid bytes of bit planes, SCALARS has
        //! NumScalars() floats, MASK has bit a set when move a is allowed. Fills PROBS (NumActions()
        //! entries) and VALUE, and returns the move: the most likely one, or a draw from the
        //! distribution when SAMPLE is set (RNG is advanced then).
        int Decide( uint8_t const * local, uint8_t const * global, float const * scalars, unsigned mask,
                    bool sample, float * probs, float * value, uint32_t & rng ) const;

        int Grid() const { return grid_; }
        int NumScalars() const { return nScalars_; }
        int NumActions() const { return nActions_; }

    private:
        void Tower( std::vector< Conv > const & convs, Linear const & fc, int planes, uint8_t const * packed,
                    std::vector< float > & out ) const;

        bool loaded_;
        int grid_, localPlanes_, globalPlanes_, nScalars_, nActions_, update_;
        std::vector< Conv > local_, global_;
        Linear localFc_, globalFc_, scalars0_, scalars2_, trunk0_, trunk2_, pi_, v_;
    };
}

#endif
