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

#include "gNeuralNet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <istream>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

namespace
{
// ------------------------------------------------------------------ file format
// written by ai/arma_ai/export.py:
//   char magic[8] = "ARMABRN1" (v1) or "ARMABRN2" (v2)
//   u32 grid, localPlanes, globalPlanes, nScalars, nActions, update, dtype (16 or 32), nLayers
//   v2 only, then: u32 nMaps, u32 planes[nMaps], u32 stackPrev, u32 nTowers, per tower u32 k, u32 mapIdx[k]
//   per layer: u8 kind (1 conv, 2 linear), u8 nameLen, name,
//              conv: u32 cout, cin, k, stride, pad; weights [cout*cin*k*k]; bias [cout]
//              linear: u32 out, in; weights [out*in]; bias [out]
//   weights are little-endian float16 (dtype 16) or float32 (dtype 32)
// Layer names: v1 "local.net.N", "globl.net.N", "local.fc", "globl.fc"; v2 "towers.M.convs.N",
// "towers.M.res.B.0" / ".1" (a residual pair), "towers.M.fc"; both: "scalars.0", "scalars.2",
// "trunk.0", "trunk.2", "pi", "v".

bool ReadU32( std::istream & in, uint32_t & v )
{
    unsigned char b[4];
    if ( !in.read( reinterpret_cast< char * >( b ), 4 ) )
        return false;
    v = uint32_t( b[0] ) | uint32_t( b[1] ) << 8 | uint32_t( b[2] ) << 16 | uint32_t( b[3] ) << 24;
    return true;
}

float HalfToFloat( uint16_t h )
{
    uint32_t sign = uint32_t( h & 0x8000 ) << 16;
    uint32_t exp = ( h >> 10 ) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if ( exp == 0 )
    {
        if ( mant == 0 )
            bits = sign;
        else
        {
            exp = 127 - 15 + 1;
            while ( !( mant & 0x400 ) )
            {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ff;
            bits = sign | ( exp << 23 ) | ( mant << 13 );
        }
    }
    else if ( exp == 31 )
        bits = sign | 0x7f800000 | ( mant << 13 );
    else
        bits = sign | ( ( exp + 127 - 15 ) << 23 ) | ( mant << 13 );
    float f;
    memcpy( &f, &bits, 4 );
    return f;
}

bool ReadFloats( std::istream & in, uint32_t dtype, size_t n, std::vector< float > & out )
{
    out.resize( n );
    if ( dtype == 32 )
    {
        std::vector< unsigned char > raw( n * 4 );
        if ( n && !in.read( reinterpret_cast< char * >( &raw[0] ), raw.size() ) )
            return false;
        for ( size_t i = 0; i < n; ++i )
        {
            uint32_t bits = uint32_t( raw[4 * i] ) | uint32_t( raw[4 * i + 1] ) << 8 | uint32_t( raw[4 * i + 2] ) << 16 | uint32_t( raw[4 * i + 3] ) << 24;
            memcpy( &out[i], &bits, 4 );
        }
    }
    else
    {
        std::vector< unsigned char > raw( n * 2 );
        if ( n && !in.read( reinterpret_cast< char * >( &raw[0] ), raw.size() ) )
            return false;
        for ( size_t i = 0; i < n; ++i )
            out[i] = HalfToFloat( uint16_t( raw[2 * i] ) | uint16_t( raw[2 * i + 1] ) << 8 );
    }
    return true;
}

inline float Relu( float x ) { return x > 0 ? x : 0; }

// ------------------------------------------------------------------ matrix multiply
typedef void ( *SgemmFn )( int order, int transA, int transB, int M, int N, int K, float alpha,
                           float const * A, int lda, float const * B, int ldb, float beta, float * C, int ldc );

SgemmFn FindSgemm()
{
#ifdef __APPLE__
    void * lib = dlopen( "/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LAZY );
    if ( lib )
        return reinterpret_cast< SgemmFn >( dlsym( lib, "cblas_sgemm" ) );
#endif
    return NULL;
}

void Matmul( float const * A, float const * B, float * C, int M, int N, int K )
{
    static SgemmFn sgemm = FindSgemm();
    if ( sgemm )
    {
        sgemm( 101, 111, 111, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N );
        return;
    }
    for ( int m = 0; m < M; ++m )
    {
        float * __restrict c = C + size_t( m ) * N;
        for ( int n = 0; n < N; ++n )
            c[n] = 0;
        for ( int k = 0; k < K; ++k )
        {
            float const a = A[size_t( m ) * K + k];
            float const * __restrict b = B + size_t( k ) * N;
            for ( int n = 0; n < N; ++n )
                c[n] += a * b[n];
        }
    }
}

// ------------------------------------------------------------------ layers
//! out = conv(in) + bias (no activation); in is [cin][hin][win], out becomes [cout][hout][wout]
void RunConvRaw( gNeuralNet::Conv const & L, std::vector< float > const & in, int hin, int win,
                 std::vector< float > & out, int & hout, int & wout )
{
    hout = ( hin + 2 * L.pad - L.k ) / L.stride + 1;
    wout = ( win + 2 * L.pad - L.k ) / L.stride + 1;
    int const K = L.cin * L.k * L.k, N = hout * wout;
    static std::vector< float > cols;
    cols.assign( size_t( K ) * N, 0.0f );
    for ( int ci = 0; ci < L.cin; ++ci )
    {
        float const * inPlane = &in[size_t( ci ) * hin * win];
        for ( int ky = 0; ky < L.k; ++ky )
            for ( int kx = 0; kx < L.k; ++kx )
            {
                float * row = &cols[( size_t( ci ) * L.k * L.k + ky * L.k + kx ) * N];
                for ( int oy = 0; oy < hout; ++oy )
                {
                    int const iy = oy * L.stride - L.pad + ky;
                    if ( iy < 0 || iy >= hin )
                        continue;
                    for ( int ox = 0; ox < wout; ++ox )
                    {
                        int const ix = ox * L.stride - L.pad + kx;
                        if ( ix >= 0 && ix < win )
                            row[oy * wout + ox] = inPlane[iy * win + ix];
                    }
                }
            }
    }
    out.resize( size_t( L.cout ) * N );
    Matmul( &L.w[0], &cols[0], &out[0], L.cout, N, K );
    for ( int co = 0; co < L.cout; ++co )
    {
        float * o = &out[size_t( co ) * N];
        for ( int i = 0; i < N; ++i )
            o[i] += L.b[co];
    }
}

void RunLinear( gNeuralNet::Linear const & L, float const * in, std::vector< float > & out )
{
    out.resize( L.out );
    Matmul( &L.w[0], in, &out[0], L.out, 1, L.in );
    for ( int o = 0; o < L.out; ++o )
        out[o] += L.b[o];
}

void ReluInPlace( std::vector< float > & v )
{
    for ( size_t i = 0; i < v.size(); ++i )
        v[i] = Relu( v[i] );
}

uint32_t NextRandom( uint32_t & s )
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

bool StartsWith( std::string const & s, char const * p )
{
    return s.compare( 0, strlen( p ), p ) == 0;
}
} // namespace

// ------------------------------------------------------------------ Net
gNeuralNet::Net::Net()
    : loaded_( false ), version_( 0 ), grid_( 0 ), nMaps_( 0 ), nScalars_( 0 ), nActions_( 0 ), update_( 0 ),
      stackPrev_( false ), hasAux_( false ), havePrev_( false )
{
}

bool gNeuralNet::Net::Load( std::istream & in, std::string & error )
{
    loaded_ = false;
    towers_.clear();
    mapPlanes_.clear();
    havePrev_ = false;
    hasAux_ = false;
    char magic[8];
    if ( !in.read( magic, 8 ) || memcmp( magic, "ARMABRN", 7 ) != 0 || ( magic[7] != '1' && magic[7] != '2' ) )
    {
        error = "not a policy file (bad magic)";
        return false;
    }
    version_ = magic[7] - '0';
    uint32_t grid, lp, gp, ns, na, update, dtype, nLayers;
    if ( !ReadU32( in, grid ) || !ReadU32( in, lp ) || !ReadU32( in, gp ) || !ReadU32( in, ns ) ||
         !ReadU32( in, na ) || !ReadU32( in, update ) || !ReadU32( in, dtype ) || !ReadU32( in, nLayers ) )
    {
        error = "truncated header";
        return false;
    }
    if ( ( dtype != 16 && dtype != 32 ) || grid == 0 || grid > 256 || lp > 8 || gp > 8 || na == 0 || na > 16 || nLayers > 256 )
    {
        error = "unsupported policy file";
        return false;
    }
    grid_ = int( grid );
    nScalars_ = int( ns );
    nActions_ = int( na );
    update_ = int( update );
    if ( version_ == 1 )
    {
        // callers always pass the game's four maps (local, close, global, territory); a v1 net
        // reads the local one and the global one
        nMaps_ = 4;
        mapPlanes_.push_back( int( lp ) );
        mapPlanes_.push_back( int( lp ) );
        mapPlanes_.push_back( int( gp ) );
        mapPlanes_.push_back( int( gp ) );
        stackPrev_ = false;
        towers_.resize( 2 );
        towers_[0].maps.push_back( 0 );
        towers_[1].maps.push_back( 2 );
    }
    else
    {
        uint32_t nMaps, stack;
        if ( !ReadU32( in, nMaps ) || nMaps == 0 || nMaps > 8 )
        {
            error = "bad map count";
            return false;
        }
        nMaps_ = int( nMaps );
        for ( uint32_t m = 0; m < nMaps; ++m )
        {
            uint32_t p;
            if ( !ReadU32( in, p ) || p == 0 || p > 8 )
            {
                error = "bad plane count";
                return false;
            }
            mapPlanes_.push_back( int( p ) );
        }
        if ( !ReadU32( in, stack ) )
        {
            error = "truncated header";
            return false;
        }
        stackPrev_ = stack != 0;
        uint32_t nTowers;
        if ( !ReadU32( in, nTowers ) || nTowers == 0 || nTowers > 8 )
        {
            error = "bad tower count";
            return false;
        }
        towers_.resize( nTowers );
        for ( uint32_t t = 0; t < nTowers; ++t )
        {
            uint32_t k;
            if ( !ReadU32( in, k ) || k == 0 || k > nMaps )
            {
                error = "bad tower table";
                return false;
            }
            for ( uint32_t j = 0; j < k; ++j )
            {
                uint32_t m;
                if ( !ReadU32( in, m ) || m >= nMaps )
                {
                    error = "bad tower table";
                    return false;
                }
                towers_[t].maps.push_back( int( m ) );
            }
        }
    }

    bool haveS0 = false, haveS2 = false, haveT0 = false, haveT2 = false, havePi = false, haveV = false;
    for ( uint32_t l = 0; l < nLayers; ++l )
    {
        unsigned char kind, nameLen;
        if ( !in.read( reinterpret_cast< char * >( &kind ), 1 ) || !in.read( reinterpret_cast< char * >( &nameLen ), 1 ) )
        {
            error = "truncated layer header";
            return false;
        }
        std::string name( nameLen, '\0' );
        if ( nameLen && !in.read( &name[0], nameLen ) )
        {
            error = "truncated layer name";
            return false;
        }
        if ( kind == 1 )
        {
            Conv c;
            uint32_t cout, cin, k, stride, pad;
            if ( !ReadU32( in, cout ) || !ReadU32( in, cin ) || !ReadU32( in, k ) || !ReadU32( in, stride ) || !ReadU32( in, pad ) )
            {
                error = "truncated conv " + name;
                return false;
            }
            c.cout = int( cout );
            c.cin = int( cin );
            c.k = int( k );
            c.stride = int( stride );
            c.pad = int( pad );
            if ( !ReadFloats( in, dtype, size_t( cout ) * cin * k * k, c.w ) || !ReadFloats( in, dtype, cout, c.b ) )
            {
                error = "truncated conv weights " + name;
                return false;
            }
            int tower = -1, res = 0;
            if ( version_ == 1 )
            {
                if ( StartsWith( name, "local.net." ) ) tower = 0;
                else if ( StartsWith( name, "globl.net." ) ) tower = 1;
            }
            else if ( StartsWith( name, "towers." ) )
            {
                // towers.M.convs.N or towers.M.res.B.0 / .1
                tower = atoi( name.c_str() + 7 );
                size_t dot = name.find( '.', 7 );
                std::string rest = dot == std::string::npos ? "" : name.substr( dot + 1 );
                if ( StartsWith( rest, "res." ) )
                    res = rest[rest.size() - 1] == '0' ? 1 : 2;
            }
            if ( tower < 0 || tower >= int( towers_.size() ) )
            {
                error = "unexpected conv layer " + name;
                return false;
            }
            towers_[tower].convs.push_back( c );
            towers_[tower].resPair.push_back( res );
        }
        else if ( kind == 2 )
        {
            Linear L;
            uint32_t out, inN;
            if ( !ReadU32( in, out ) || !ReadU32( in, inN ) )
            {
                error = "truncated linear " + name;
                return false;
            }
            L.out = int( out );
            L.in = int( inN );
            if ( !ReadFloats( in, dtype, size_t( out ) * inN, L.w ) || !ReadFloats( in, dtype, out, L.b ) )
            {
                error = "truncated linear weights " + name;
                return false;
            }
            if ( name == "local.fc" ) { towers_[0].fc = L; towers_[0].hasFc = true; }
            else if ( name == "globl.fc" ) { towers_[1].fc = L; towers_[1].hasFc = true; }
            else if ( version_ == 2 && StartsWith( name, "towers." ) && name.find( ".fc" ) != std::string::npos )
            {
                int t = atoi( name.c_str() + 7 );
                if ( t < 0 || t >= int( towers_.size() ) )
                {
                    error = "unexpected linear layer " + name;
                    return false;
                }
                towers_[t].fc = L;
                towers_[t].hasFc = true;
            }
            else if ( name == "scalars.0" ) { scalars0_ = L; haveS0 = true; }
            else if ( name == "scalars.2" ) { scalars2_ = L; haveS2 = true; }
            else if ( name == "trunk.0" ) { trunk0_ = L; haveT0 = true; }
            else if ( name == "trunk.2" ) { trunk2_ = L; haveT2 = true; }
            else if ( name == "pi" ) { pi_ = L; havePi = true; }
            else if ( name == "v" ) { v_ = L; haveV = true; }
            else if ( name == "aux" ) { aux_ = L; hasAux_ = true; }
            else
            {
                error = "unexpected linear layer " + name;
                return false;
            }
        }
        else
        {
            error = "unknown layer kind";
            return false;
        }
    }
    for ( size_t t = 0; t < towers_.size(); ++t )
        if ( towers_[t].convs.empty() || !towers_[t].hasFc )
        {
            error = "policy file is missing a tower";
            return false;
        }
    if ( !haveS0 || !haveS2 || !haveT0 || !haveT2 || !havePi || !haveV )
    {
        error = "policy file is missing layers";
        return false;
    }
    int fused = scalars2_.out;
    for ( size_t t = 0; t < towers_.size(); ++t )
        fused += towers_[t].fc.out;
    if ( pi_.out != nActions_ || scalars0_.in != nScalars_ || trunk0_.in != fused )
    {
        error = "policy file layers do not fit together";
        return false;
    }
    prev_.assign( nMaps_, std::vector< uint8_t >( size_t( grid_ ) * grid_, 0 ) );
    loaded_ = true;
    return true;
}

void gNeuralNet::Net::Unpack( uint8_t const * packed, int planes, std::vector< float > & out, size_t offsetPlanes, size_t totalPlanes ) const
{
    int const g = grid_;
    if ( out.size() != totalPlanes * g * g )
        out.assign( totalPlanes * g * g, 0.0f );
    for ( int p = 0; p < planes; ++p )
    {
        float * plane = &out[( offsetPlanes + p ) * g * g];
        for ( int i = 0; i < g * g; ++i )
            plane[i] = ( packed[i] >> p ) & 1 ? 1.0f : 0.0f;
    }
}

void gNeuralNet::Net::RunTower( Tower const & t, std::vector< float > & x, int h, int w, std::vector< float > & out ) const
{
    std::vector< float > y, skip;
    for ( size_t l = 0; l < t.convs.size(); ++l )
    {
        int ho, wo;
        int res = t.resPair[l];
        if ( res == 1 )
            skip = x;  // keep the block input for the skip connection
        RunConvRaw( t.convs[l], x, h, w, y, ho, wo );
        if ( res == 2 )
            for ( size_t i = 0; i < y.size(); ++i )
                y[i] += skip[i];
        ReluInPlace( y );
        x.swap( y );
        h = ho;
        w = wo;
    }
    RunLinear( t.fc, &x[0], out );
    ReluInPlace( out );
}

int gNeuralNet::Net::Decide( uint8_t const * const * maps, float const * scalars, unsigned mask,
                             bool sample, float * probs, float * value, uint32_t & rng )
{
    std::vector< float > h, part, t, x;
    for ( size_t ti = 0; ti < towers_.size(); ++ti )
    {
        Tower const & tw = towers_[ti];
        // the tower's input: its maps' current planes in order, then (v2) the same maps' previous planes
        size_t cur = 0;
        for ( size_t j = 0; j < tw.maps.size(); ++j )
            cur += mapPlanes_[tw.maps[j]];
        size_t total = stackPrev_ ? 2 * cur : cur;
        x.clear();
        size_t off = 0;
        for ( size_t j = 0; j < tw.maps.size(); ++j )
        {
            int m = tw.maps[j];
            Unpack( maps[m], mapPlanes_[m], x, off, total );
            off += mapPlanes_[m];
        }
        if ( stackPrev_ )
            for ( size_t j = 0; j < tw.maps.size(); ++j )
            {
                int m = tw.maps[j];
                Unpack( havePrev_ ? &prev_[m][0] : maps[m], mapPlanes_[m], x, off, total );
                off += mapPlanes_[m];
            }
        RunTower( tw, x, grid_, grid_, part );
        h.insert( h.end(), part.begin(), part.end() );
    }
    if ( stackPrev_ )
    {
        for ( int m = 0; m < nMaps_; ++m )
            memcpy( &prev_[m][0], maps[m], size_t( grid_ ) * grid_ );
        havePrev_ = true;
    }
    RunLinear( scalars0_, scalars, t );
    ReluInPlace( t );
    RunLinear( scalars2_, &t[0], part );
    ReluInPlace( part );
    h.insert( h.end(), part.begin(), part.end() );

    RunLinear( trunk0_, &h[0], t );
    ReluInPlace( t );
    RunLinear( trunk2_, &t[0], h );
    ReluInPlace( h );

    std::vector< float > logits, val;
    RunLinear( pi_, &h[0], logits );
    RunLinear( v_, &h[0], val );
    if ( value )
        *value = val[0];

    float best = -1e30f;
    for ( int a = 0; a < nActions_; ++a )
    {
        if ( !( ( mask >> a ) & 1 ) )
            logits[a] = -1e8f;
        best = std::max( best, logits[a] );
    }
    float sum = 0;
    for ( int a = 0; a < nActions_; ++a )
    {
        logits[a] = std::exp( logits[a] - best );
        sum += logits[a];
    }
    int pick = 0;
    for ( int a = 0; a < nActions_; ++a )
    {
        logits[a] /= sum;
        if ( probs )
            probs[a] = logits[a];
        if ( logits[a] > logits[pick] )
            pick = a;
    }
    if ( sample )
    {
        float r = ( NextRandom( rng ) >> 8 ) * ( 1.0f / 16777216.0f ), acc = 0;
        for ( int a = 0; a < nActions_; ++a )
        {
            acc += logits[a];
            if ( r < acc && ( ( mask >> a ) & 1 ) )
                return a;
        }
    }
    return pick;
}
