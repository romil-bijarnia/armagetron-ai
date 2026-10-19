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
//! one of the policy's moves. Plain C++ on purpose, so it has no dependency on the rest of the
//! engine and can be checked against PyTorch on its own.
//!
//! Loads the V4 policy: separate local scales, arena-conditioned features, opponent attention
//! and six moves. The four maps are local, close, global and territory, with the previous frame
//! stacked beneath them. Observation layout 2 has 108 scalars and 12 values per opponent token.
namespace gNeuralNet
{
    struct Conv
    {
        int cout, cin, k, stride, pad;
        std::vector< float > w; //!< [cout][cin][k][k]
        std::vector< float > b;
        Conv(): cout( 0 ), cin( 0 ), k( 0 ), stride( 0 ), pad( 0 ) {}
    };

    struct Linear
    {
        int out, in;
        std::vector< float > w; //!< [out][in]
        std::vector< float > b;
        Linear(): out( 0 ), in( 0 ) {}
    };

    struct LayerNorm
    {
        int n;
        std::vector< float > w, b;
        LayerNorm(): n( 0 ) {}
    };

    //! One residual block; a pooling block also feeds the board-wide mean and max of its first
    //! GCH channels back as a per-channel bias (KataGo's global pooling)
    struct Block
    {
        Conv c0, c1;
        bool gpool;
        int gch;
        Linear gp;
        bool haveC0, haveC1;
        Block(): gpool( false ), gch( 0 ), haveC0( false ), haveC1( false ) {}
    };

    //! One pre-norm attention layer over the map's cells
    struct Attn
    {
        LayerNorm ln1, ln2;
        Linear qkv, proj, fc1, fc2;
        int have;
        Attn(): have( 0 ) {}
    };

    //! One tower: a stem conv, residual blocks, optional arena attention, two strided convs, a linear.
    //! Local towers also keep a residual block and reduction between the stem and blocks.
    struct Tower
    {
        std::vector< int > maps;    //!< which maps feed it, in the order their planes are stacked
        Linear fc;
        bool hasFc;
        Conv stem, down0, down1;
        bool haveStem, haveDown0, haveDown1;
        std::vector< Block > blocks;
        std::vector< float > pos;   //!< [cells][width], the attention's position embedding
        std::vector< Attn > attn;
        Block pre;
        Conv reduce;
        bool haveReduce;
        Tower(): hasFc( false ), haveStem( false ), haveDown0( false ), haveDown1( false ), haveReduce( false ) {}
    };

    class Net
    {
    public:
        Net();

        //! reads a policy file; on failure returns false and says why in ERROR
        bool Load( std::istream & in, std::string & error );
        bool Loaded() const { return loaded_; }
        int Update() const { return update_; } //!< the training update the weights come from
        int Version() const { return loaded_ ? 4 : 0; }

        //! Previous maps plus the 256-value GRU state after consuming the last observation.
        //! Keep one per driven cycle, never share it between cycles, and
        //! Reset() it when the cycle's life starts.
        struct Memory
        {
            bool have;
            std::vector< std::vector< uint8_t > > maps;
            std::vector< float > recurrent;
            Memory(): have( false ) {}
            void Reset() { have = false; maps.clear(); recurrent.clear(); }
        };

        //! one decision. MAPS points at NumMaps() grids of Grid()*Grid() bytes, the game's four
        //! maps in order: local, close, global, territory. SCALARS has NumScalars() floats, MASK has
        //! bit a set when move a is allowed. Fills PROBS (NumActions() entries) and VALUE, and
        //! returns the move: the most likely one, or a draw from the distribution when SAMPLE is
        //! set (RNG is advanced then). MEMORY is the driven cycle's own (see Memory).
        int Decide( uint8_t const * const * maps, float const * scalars, unsigned mask,
                    bool sample, float * probs, float * value, uint32_t & rng, Memory & memory,
                    float const * context = NULL );

        int Grid() const { return grid_; }
        int NumMaps() const { return nMaps_; }
        int NumScalars() const { return nScalars_; }
        int NumActions() const { return nActions_; }

    private:
        void RunTowerFeatures( Tower const & t, std::vector< float > & x, int & h, int & w ) const;
        void RunTowerHead( Tower const & t, std::vector< float > const & x, int h, int w, std::vector< float > & out ) const;
        void TowerInput( Tower const & t, uint8_t const * const * maps, Memory const & memory, std::vector< float > & x ) const;
        void Unpack( uint8_t const * packed, int planes, std::vector< float > & out, size_t offsetPlanes, size_t totalPlanes ) const;

        bool loaded_;
        int grid_, nMaps_, nScalars_, nActions_, update_;
        std::vector< int > mapPlanes_;      //!< eight planes per map
        std::vector< Tower > towers_;       //!< local, close, and combined global/territory towers
        Linear scalars0_, scalars2_, trunk0_, trunk2_, pi_, v_, aux_;
        Linear conditioning_[2], opponents0_, opponents2_, opponentQuery_;
        Linear memoryInput_, memoryHidden_, memoryProjection_;
        bool hasAux_;
    };
}

#endif
