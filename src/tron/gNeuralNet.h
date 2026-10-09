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
//! observation (the maps, and the exact numbers the neural bridge sends to the trainer) into
//! one of the four moves. Plain C++ on purpose, so it has no dependency on the rest of the
//! engine and can be checked against PyTorch on its own.
//!
//! Two generations of network are understood: v1 (two maps, plain conv towers) and v2 (four
//! maps: local, close, global, territory; residual towers; the previous frame's maps stacked).
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

    //! one tower: a stem conv, residual blocks (pairs of convs with a skip), two strided convs, a linear
    struct Tower
    {
        std::vector< Conv > convs;  //!< in forward order; residual pairs are flagged by `resPair`
        std::vector< int > resPair; //!< for each conv: 0 = plain, 1 = first of a residual pair, 2 = second
        std::vector< int > maps;    //!< which maps feed it, in the order their planes are stacked
        Linear fc;
        bool hasFc;
        Tower(): hasFc( false ) {}
    };

    class Net
    {
    public:
        Net();

        //! reads a policy file; on failure returns false and says why in ERROR
        bool Load( std::istream & in, std::string & error );
        bool Loaded() const { return loaded_; }
        int Update() const { return update_; } //!< the training update the weights come from
        int Version() const { return version_; } //!< 1 or 2

        //! one decision. MAPS points at NumMaps() grids of Grid()*Grid() bytes, the game's four
        //! maps in order: local, close, global, territory (a v1 net reads only local and global). SCALARS has NumScalars() floats, MASK has
        //! bit a set when move a is allowed. Fills PROBS (NumActions() entries) and VALUE, and
        //! returns the move: the most likely one, or a draw from the distribution when SAMPLE is
        //! set (RNG is advanced then). A v2 net keeps the previous frame's maps itself (frame
        //! stacking); call NewLife() when the cycle it drives respawns.
        int Decide( uint8_t const * const * maps, float const * scalars, unsigned mask,
                    bool sample, float * probs, float * value, uint32_t & rng );

        //! forget the previous frame (a new round or a new cycle)
        void NewLife() { havePrev_ = false; }

        int Grid() const { return grid_; }
        int NumMaps() const { return nMaps_; }
        int NumScalars() const { return nScalars_; }
        int NumActions() const { return nActions_; }

    private:
        void RunTower( Tower const & t, std::vector< float > & x, int h, int w, std::vector< float > & out ) const;
        void Unpack( uint8_t const * packed, int planes, std::vector< float > & out, size_t offsetPlanes, size_t totalPlanes ) const;

        bool loaded_;
        int version_, grid_, nMaps_, nScalars_, nActions_, update_;
        std::vector< int > mapPlanes_;      //!< planes per map (v1: 8, 8; v2: 8, 8, 8, 8)
        bool stackPrev_;                    //!< v2: the previous frame's maps are stacked under the current ones
        std::vector< Tower > towers_;       //!< v1: one per map; v2: as the file's tower table says
        Linear scalars0_, scalars2_, trunk0_, trunk2_, pi_, v_, aux_;
        bool hasAux_;
        // frame stacking state
        bool havePrev_;
        std::vector< std::vector< uint8_t > > prev_;
    };
}

#endif
