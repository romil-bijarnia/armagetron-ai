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
//   char magic[8] = "ARMABRN1"
//   u32 grid, localPlanes, globalPlanes, nScalars, nActions, update, dtype (16 or 32), nLayers
//   per layer: u8 kind (1 conv, 2 linear), u8 nameLen, name,
//              conv: u32 cout, cin, k, stride, pad; weights [cout*cin*k*k]; bias [cout]
//              linear: u32 out, in; weights [out*in]; bias [out]
//   weights are little-endian float16 (dtype 16) or float32 (dtype 32)

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
            // subnormal: normalise it
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
// C[M][N] = A[M][K] * B[K][N] (row major). On a Mac the work goes to Accelerate's cblas_sgemm,
// found at run time so the build needs no new flags; elsewhere the plain loops below run.
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
        sgemm( 101 /* row major */, 111 /* no trans */, 111, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N );
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
//! out = relu(conv(in)); in is [cin][hin][win], out becomes [cout][hout][wout]
void RunConv( gNeuralNet::Conv const & L, std::vector< float > const & in, int hin, int win,
              std::vector< float > & out, int & hout, int & wout )
{
    hout = ( hin + 2 * L.pad - L.k ) / L.stride + 1;
    wout = ( win + 2 * L.pad - L.k ) / L.stride + 1;
    int const K = L.cin * L.k * L.k, N = hout * wout;
    // im2col: every column holds the input patch one output cell looks at
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
            o[i] = Relu( o[i] + L.b[co] );
    }
}

//! out = W in + b (no activation)
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
    // xorshift32
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
} // namespace

// ------------------------------------------------------------------ Net
gNeuralNet::Net::Net()
    : loaded_( false ), grid_( 0 ), localPlanes_( 0 ), globalPlanes_( 0 ), nScalars_( 0 ), nActions_( 0 ), update_( 0 )
{
}

bool gNeuralNet::Net::Load( std::istream & in, std::string & error )
{
    loaded_ = false;
    local_.clear();
    global_.clear();
    char magic[8];
    if ( !in.read( magic, 8 ) || memcmp( magic, "ARMABRN1", 8 ) != 0 )
    {
        error = "not a policy file (bad magic)";
        return false;
    }
    uint32_t grid, lp, gp, ns, na, update, dtype, nLayers;
    if ( !ReadU32( in, grid ) || !ReadU32( in, lp ) || !ReadU32( in, gp ) || !ReadU32( in, ns ) ||
         !ReadU32( in, na ) || !ReadU32( in, update ) || !ReadU32( in, dtype ) || !ReadU32( in, nLayers ) )
    {
        error = "truncated header";
        return false;
    }
    if ( ( dtype != 16 && dtype != 32 ) || grid == 0 || grid > 256 || lp > 8 || gp > 8 || na == 0 || na > 16 || nLayers > 64 )
    {
        error = "unsupported policy file";
        return false;
    }
    grid_ = int( grid );
    localPlanes_ = int( lp );
    globalPlanes_ = int( gp );
    nScalars_ = int( ns );
    nActions_ = int( na );
    update_ = int( update );

    bool haveLocalFc = false, haveGlobalFc = false, haveS0 = false, haveS2 = false, haveT0 = false, haveT2 = false,
         havePi = false, haveV = false;
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
            if ( name.compare( 0, 10, "local.net." ) == 0 )
                local_.push_back( c );
            else if ( name.compare( 0, 10, "globl.net." ) == 0 )
                global_.push_back( c );
            else
            {
                error = "unexpected conv layer " + name;
                return false;
            }
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
            if ( name == "local.fc" ) { localFc_ = L; haveLocalFc = true; }
            else if ( name == "globl.fc" ) { globalFc_ = L; haveGlobalFc = true; }
            else if ( name == "scalars.0" ) { scalars0_ = L; haveS0 = true; }
            else if ( name == "scalars.2" ) { scalars2_ = L; haveS2 = true; }
            else if ( name == "trunk.0" ) { trunk0_ = L; haveT0 = true; }
            else if ( name == "trunk.2" ) { trunk2_ = L; haveT2 = true; }
            else if ( name == "pi" ) { pi_ = L; havePi = true; }
            else if ( name == "v" ) { v_ = L; haveV = true; }
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
    if ( local_.empty() || global_.empty() || !haveLocalFc || !haveGlobalFc || !haveS0 || !haveS2 || !haveT0 || !haveT2 || !havePi || !haveV )
    {
        error = "policy file is missing layers";
        return false;
    }
    if ( pi_.out != nActions_ || scalars0_.in != nScalars_ || trunk0_.in != localFc_.out + globalFc_.out + scalars2_.out )
    {
        error = "policy file layers do not fit together";
        return false;
    }
    loaded_ = true;
    return true;
}

void gNeuralNet::Net::Tower( std::vector< Conv > const & convs, Linear const & fc, int planes, uint8_t const * packed,
                             std::vector< float > & out ) const
{
    // unpack the bit planes: plane p of cell i is bit p of byte i
    int const g = grid_;
    std::vector< float > a( size_t( planes ) * g * g ), b;
    for ( int p = 0; p < planes; ++p )
    {
        float * plane = &a[size_t( p ) * g * g];
        for ( int i = 0; i < g * g; ++i )
            plane[i] = ( packed[i] >> p ) & 1 ? 1.0f : 0.0f;
    }
    int h = g, w = g;
    for ( size_t l = 0; l < convs.size(); ++l )
    {
        int ho, wo;
        RunConv( convs[l], a, h, w, b, ho, wo );
        a.swap( b );
        h = ho;
        w = wo;
    }
    RunLinear( fc, &a[0], out );
    ReluInPlace( out );
}

int gNeuralNet::Net::Decide( uint8_t const * local, uint8_t const * global, float const * scalars, unsigned mask,
                             bool sample, float * probs, float * value, uint32_t & rng ) const
{
    std::vector< float > hl, hg, hs, t;
    Tower( local_, localFc_, localPlanes_, local, hl );
    Tower( global_, globalFc_, globalPlanes_, global, hg );
    RunLinear( scalars0_, scalars, t );
    ReluInPlace( t );
    RunLinear( scalars2_, &t[0], hs );
    ReluInPlace( hs );

    std::vector< float > h;
    h.reserve( hl.size() + hg.size() + hs.size() );
    h.insert( h.end(), hl.begin(), hl.end() );
    h.insert( h.end(), hg.begin(), hg.end() );
    h.insert( h.end(), hs.begin(), hs.end() );
    RunLinear( trunk0_, &h[0], t );
    ReluInPlace( t );
    RunLinear( trunk2_, &t[0], h );
    ReluInPlace( h );

    std::vector< float > logits, val;
    RunLinear( pi_, &h[0], logits );
    RunLinear( v_, &h[0], val );
    if ( value )
        *value = val[0];

    // the same masking and softmax as the trainer
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
