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
#include <set>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

namespace
{
// ------------------------------------------------------------------ file format
// written by ai/arma_ai/export.py:
//   char magic[8] = "ARMABRN4"
//   u32 grid, localPlanes, globalPlanes, nScalars, nActions, update, dtype (16 or 32), nLayers
//   u32 nMaps, u32 planes[nMaps], u32 stackPrev, u32 nTowers, per tower u32 k, u32 mapIdx[k]
//   per layer: u8 kind (1 conv, 2 linear, 3 layer norm, 4 parameter, 5 GRU), u8 nameLen, name,
//              conv: u32 cout, cin, k, stride, pad; weights [cout*cin*k*k]; bias [cout]
//              linear: u32 out, in; weights [out*in]; bias [out]
//              layer norm: u32 n; weight [n]; bias [n]
//              parameter: u32 ndim, dims[ndim]; values
//              GRU: u32 input, hidden; weight_ih, weight_hh, bias_ih, bias_hh (r,z,n)
//   weights are little-endian float16 (dtype 16) or float32 (dtype 32)
// Tower names: "towers.M.stem", "towers.M.res.B.c0/.gp/.c1", "towers.M.pos",
// "towers.M.attn.L.*", "towers.M.down0/1", "towers.M.fc", and local "pre.c0/.c1", "reduce".
// Other layers: "conditioning.0/.1", "opponents.0/.2", "opponent_query", "scalars.0/.2",
// "trunk.0/.2", "pi", "v", "aux". The value head is bounded with tanh.

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

void MatmulPortable( float const * A, float const * B, float * C, int M, int N, int K,
                     bool transposeA, bool transposeB )
{
    for ( int m = 0; m < M; ++m )
    {
        float * __restrict c = C + size_t( m ) * N;
        for ( int n = 0; n < N; ++n )
            c[n] = 0;
        for ( int k = 0; k < K; ++k )
        {
            float const a = transposeA ? A[size_t( k ) * M + m] : A[size_t( m ) * K + k];
            if ( transposeB )
            {
                for ( int n = 0; n < N; ++n )
                    c[n] += a * B[size_t( n ) * K + k];
            }
            else
            {
                float const * __restrict b = B + size_t( k ) * N;
                for ( int n = 0; n < N; ++n )
                    c[n] += a * b[n];
            }
        }
    }
}

// M, N and K describe op(A) * op(B); operands and output use row-major storage.
void Matmul( float const * A, float const * B, float * C, int M, int N, int K,
             bool transposeA = false, bool transposeB = false )
{
    static SgemmFn sgemm = FindSgemm();
    if ( sgemm )
    {
        sgemm( 101, transposeA ? 112 : 111, transposeB ? 112 : 111, M, N, K, 1.0f,
               A, transposeA ? M : K, B, transposeB ? K : N, 0.0f, C, N );
        return;
    }
    MatmulPortable( A, B, C, M, N, K, transposeA, transposeB );
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

inline float Gelu( float x )
{
    return 0.5f * x * ( 1.0f + std::erf( x * 0.70710678118654752f ) );
}

//! layer norm over the channels of every cell; x and out are [C][N] (channel-major)
void LayerNormCells( gNeuralNet::LayerNorm const & L, std::vector< float > const & x, int C, int N, std::vector< float > & out )
{
    out.resize( x.size() );
    for ( int i = 0; i < N; ++i )
    {
        float mean = 0;
        for ( int c = 0; c < C; ++c )
            mean += x[size_t( c ) * N + i];
        mean /= C;
        float var = 0;
        for ( int c = 0; c < C; ++c )
        {
            float d = x[size_t( c ) * N + i] - mean;
            var += d * d;
        }
        float inv = 1.0f / std::sqrt( var / C + 1e-5f );
        for ( int c = 0; c < C; ++c )
            out[size_t( c ) * N + i] = ( x[size_t( c ) * N + i] - mean ) * inv * L.w[c] + L.b[c];
    }
}

//! y = W x + b for every cell: W is [out][in], x is [in][N], y becomes [out][N]
void LinearCells( gNeuralNet::Linear const & L, std::vector< float > const & x, int N, std::vector< float > & y )
{
    y.resize( size_t( L.out ) * N );
    Matmul( &L.w[0], &x[0], &y[0], L.out, N, L.in );
    for ( int o = 0; o < L.out; ++o )
    {
        float * row = &y[size_t( o ) * N];
        for ( int i = 0; i < N; ++i )
            row[i] += L.b[o];
    }
}

//! one attention layer over N cells of C channels, in place on x ([C][N])
void RunAttention( gNeuralNet::Attn const & A, std::vector< float > & x, int C, int N )
{
    int const heads = 4, dh = C / heads;
    static std::vector< float > hn, qkv, att, y, scores, normalizers;
    LayerNormCells( A.ln1, x, C, N, hn );
    LinearCells( A.qkv, hn, N, qkv );  // rows: q (C), k (C), v (C)
    att.assign( size_t( C ) * N, 0.0f );
    scores.resize( size_t( N ) * N );
    normalizers.resize( N );
    float const scale = 1.0f / std::sqrt( float( dh ) );
    for ( int h = 0; h < heads; ++h )
    {
        float const * q = &qkv[size_t( h * dh ) * N];
        float const * k = &qkv[size_t( C + h * dh ) * N];
        float const * v = &qkv[size_t( 2 * C + h * dh ) * N];
        // Q and K are channel-major [dh][N]; scores = Q^T K is [N][N].
        Matmul( q, k, &scores[0], N, N, dh, true, false );
        for ( int i = 0; i < N; ++i )
        {
            float * s = &scores[size_t( i ) * N];
            float best = -1e30f;
            for ( int j = 0; j < N; ++j )
            {
                s[j] *= scale;
                best = std::max( best, s[j] );
            }
            float sum = 0;
            for ( int j = 0; j < N; ++j )
            {
                s[j] = std::exp( s[j] - best );
                sum += s[j];
            }
            normalizers[i] = 1.0f / sum;
        }
        // Keep the channel-major result: V scores^T is [dh][N]. Apply each
        // softmax normalizer after the weighted sum, as in the scalar path.
        float * output = &att[size_t( h * dh ) * N];
        Matmul( v, &scores[0], output, dh, N, N, false, true );
        for ( int d = 0; d < dh; ++d )
            for ( int i = 0; i < N; ++i )
                output[size_t( d ) * N + i] *= normalizers[i];
    }
    LinearCells( A.proj, att, N, y );
    for ( size_t i = 0; i < x.size(); ++i )
        x[i] += y[i];
    LayerNormCells( A.ln2, x, C, N, hn );
    LinearCells( A.fc1, hn, N, y );
    for ( size_t i = 0; i < y.size(); ++i )
        y[i] = Gelu( y[i] );
    LinearCells( A.fc2, y, N, hn );
    for ( size_t i = 0; i < x.size(); ++i )
        x[i] += hn[i];
}

//! tower name parsing: "towers.M.rest" -> M and rest
bool TowerName( std::string const & name, int & tower, std::string & rest )
{
    if ( name.compare( 0, 7, "towers." ) != 0 )
        return false;
    tower = atoi( name.c_str() + 7 );
    size_t dot = name.find( '.', 7 );
    rest = dot == std::string::npos ? "" : name.substr( dot + 1 );
    return true;
}

//! "res.B.x" -> B and x; "attn.L.x" -> L and x
bool IndexedName( std::string const & rest, char const * prefix, int & index, std::string & leaf )
{
    size_t n = strlen( prefix );
    if ( rest.compare( 0, n, prefix ) != 0 )
        return false;
    index = atoi( rest.c_str() + n );
    size_t dot = rest.find( '.', n );
    leaf = dot == std::string::npos ? "" : rest.substr( dot + 1 );
    return true;
}

uint32_t NextRandom( uint32_t & s )
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

bool ConvShape( gNeuralNet::Conv const & c, int in, int out, int k, int stride, int pad )
{
    return c.cin == in && c.cout == out && c.k == k && c.stride == stride && c.pad == pad;
}

bool LinearShape( gNeuralNet::Linear const & l, int in, int out )
{
    return l.in == in && l.out == out;
}

//! v4 is a fixed architecture. Reject incomplete or mismatched files before inference indexes them.
bool ValidV4Tower( gNeuralNet::Tower const & t, bool local )
{
    int const width = local ? 64 : 128;
    if ( !t.haveStem || !t.haveDown0 || !t.haveDown1 || !t.hasFc ||
         t.blocks.size() != size_t( local ? 3 : 5 ) ||
         !ConvShape( t.down0, width, 2 * width, 3, 2, 1 ) ||
         !ConvShape( t.down1, 2 * width, 2 * width, 3, 2, 1 ) ||
         !LinearShape( t.fc, 2 * width * 16, local ? 192 : 256 ) )
        return false;
    if ( local )
    {
        if ( !ConvShape( t.stem, 16, 48, 3, 2, 1 ) || !t.pre.haveC0 || !t.pre.haveC1 || !t.haveReduce ||
             !ConvShape( t.pre.c0, 48, 48, 3, 1, 1 ) || !ConvShape( t.pre.c1, 48, 48, 3, 1, 1 ) ||
             !ConvShape( t.reduce, 48, width, 3, 2, 1 ) || !t.attn.empty() || !t.pos.empty() )
            return false;
    }
    else if ( !ConvShape( t.stem, 32, width, 4, 4, 0 ) || t.pre.haveC0 || t.pre.haveC1 || t.haveReduce ||
              t.attn.size() != 2 || t.pos.size() != size_t( 256 * width ) )
        return false;
    for ( size_t i = 0; i < t.blocks.size(); ++i )
    {
        gNeuralNet::Block const & b = t.blocks[i];
        bool const pooled = ( i + 1 ) % 3 == 0;
        if ( !b.haveC0 || !b.haveC1 || !ConvShape( b.c0, width, width, 3, 1, 1 ) ||
             !ConvShape( b.c1, width, width, 3, 1, 1 ) || b.gpool != pooled ||
             ( pooled && ( b.gch != 32 || !LinearShape( b.gp, 64, width ) ) ) )
            return false;
    }
    for ( size_t i = 0; i < t.attn.size(); ++i )
    {
        gNeuralNet::Attn const & a = t.attn[i];
        if ( a.have != 63 || a.ln1.n != width || a.ln2.n != width ||
             !LinearShape( a.qkv, width, 3 * width ) || !LinearShape( a.proj, width, width ) ||
             !LinearShape( a.fc1, width, 2 * width ) || !LinearShape( a.fc2, 2 * width, width ) )
            return false;
    }
    return true;
}
} // namespace

// ------------------------------------------------------------------ Net
gNeuralNet::Net::Net()
    : loaded_( false ), grid_( 0 ), nMaps_( 0 ), nScalars_( 0 ), nActions_( 0 ), update_( 0 ), hasAux_( false )
{
}

bool gNeuralNet::Net::Load( std::istream & in, std::string & error )
{
    loaded_ = false;
    towers_.clear();
    mapPlanes_.clear();
    hasAux_ = false;
    char magic[8];
    if ( !in.read( magic, 8 ) || memcmp( magic, "ARMABRN", 7 ) != 0 )
    {
        error = "not a policy file (bad magic)";
        return false;
    }
    if ( magic[7] != '4' )
    {
        error = "unsupported policy version; export a V4 checkpoint (ARMABRN4 required)";
        return false;
    }
    uint32_t grid, lp, gp, ns, na, update, dtype, nLayers;
    if ( !ReadU32( in, grid ) || !ReadU32( in, lp ) || !ReadU32( in, gp ) || !ReadU32( in, ns ) ||
         !ReadU32( in, na ) || !ReadU32( in, update ) || !ReadU32( in, dtype ) || !ReadU32( in, nLayers ) )
    {
        error = "truncated header";
        return false;
    }
    if ( ( dtype != 16 && dtype != 32 ) || grid == 0 || grid > 256 || lp == 0 || lp > 8 || gp == 0 || gp > 8 ||
         ns == 0 || ns > 4096 || na == 0 || na > 16 || nLayers > 256 )
    {
        error = "unsupported policy file";
        return false;
    }
    grid_ = int( grid );
    nScalars_ = int( ns );
    nActions_ = int( na );
    update_ = int( update );
    if ( nScalars_ != 108 )
    {
        error = "unsupported V4 observation layout; 108 scalars with opponent resources are required";
        return false;
    }
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

    if ( grid_ != 64 || nActions_ != 6 || nMaps_ != 4 || stack != 1 ||
         lp != 8 || gp != 8 || mapPlanes_ != std::vector< int >( 4, 8 ) || towers_.size() != 3 ||
         towers_[0].maps != std::vector< int >( 1, 0 ) || towers_[1].maps != std::vector< int >( 1, 1 ) ||
         towers_[2].maps != std::vector< int >{ 2, 3 } )
    {
        error = "unsupported v4 observation or tower layout";
        return false;
    }

    bool haveS0 = false, haveS2 = false, haveT0 = false, haveT2 = false, havePi = false, haveV = false;
    bool haveConditioning[2] = { false, false }, haveOpponents0 = false, haveOpponents2 = false, haveOpponentQuery = false;
    bool haveMemoryVersion = false, haveMemory = false, haveMemoryProjection = false;
    std::set< std::string > layerNames;
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
        if ( !layerNames.insert( name ).second )
        {
            error = "duplicate layer " + name;
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
            if ( cout == 0 || cout > 4096 || cin == 0 || cin > 4096 || k == 0 || k > 16 || stride == 0 ||
                 stride > 16 || pad > 16 || size_t( cout ) * cin * k * k > ( 1u << 26 ) )
            {
                error = "unsupported conv dimensions " + name;
                return false;
            }
            if ( !ReadFloats( in, dtype, size_t( cout ) * cin * k * k, c.w ) || !ReadFloats( in, dtype, cout, c.b ) )
            {
                error = "truncated conv weights " + name;
                return false;
            }
            int tower, b;
            std::string rest, leaf;
            if ( !TowerName( name, tower, rest ) || tower < 0 || tower >= int( towers_.size() ) )
            {
                error = "unexpected conv layer " + name;
                return false;
            }
            Tower & tw = towers_[tower];
            if ( rest == "stem" ) { tw.stem = c; tw.haveStem = true; }
            else if ( rest == "down0" ) { tw.down0 = c; tw.haveDown0 = true; }
            else if ( rest == "down1" ) { tw.down1 = c; tw.haveDown1 = true; }
            else if ( rest == "pre.c0" ) { tw.pre.c0 = c; tw.pre.haveC0 = true; }
            else if ( rest == "pre.c1" ) { tw.pre.c1 = c; tw.pre.haveC1 = true; }
            else if ( rest == "reduce" ) { tw.reduce = c; tw.haveReduce = true; }
            else if ( IndexedName( rest, "res.", b, leaf ) && b >= 0 && b < 64 && ( leaf == "c0" || leaf == "c1" ) )
            {
                if ( int( tw.blocks.size() ) <= b )
                    tw.blocks.resize( b + 1 );
                if ( leaf == "c0" ) { tw.blocks[b].c0 = c; tw.blocks[b].haveC0 = true; }
                else { tw.blocks[b].c1 = c; tw.blocks[b].haveC1 = true; }
            }
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
            if ( out == 0 || out > 16384 || inN == 0 || inN > 65536 || size_t( out ) * inN > ( 1u << 26 ) )
            {
                error = "unsupported linear dimensions " + name;
                return false;
            }
            if ( !ReadFloats( in, dtype, size_t( out ) * inN, L.w ) || !ReadFloats( in, dtype, out, L.b ) )
            {
                error = "truncated linear weights " + name;
                return false;
            }
            int tower;
            std::string rest;
            if ( TowerName( name, tower, rest ) )
            {
                if ( tower < 0 || tower >= int( towers_.size() ) )
                {
                    error = "unexpected linear layer " + name;
                    return false;
                }
                Tower & tw = towers_[tower];
                int idx;
                std::string leaf;
                if ( rest == "fc" ) { tw.fc = L; tw.hasFc = true; }
                else if ( IndexedName( rest, "res.", idx, leaf ) && leaf == "gp" && idx >= 0 && idx < 64 )
                {
                    if ( int( tw.blocks.size() ) <= idx )
                        tw.blocks.resize( idx + 1 );
                    tw.blocks[idx].gp = L;
                    tw.blocks[idx].gpool = true;
                    tw.blocks[idx].gch = L.in / 2;
                }
                else if ( IndexedName( rest, "attn.", idx, leaf ) && idx >= 0 && idx < 16 )
                {
                    if ( int( tw.attn.size() ) <= idx )
                        tw.attn.resize( idx + 1 );
                    Attn & A = tw.attn[idx];
                    if ( leaf == "qkv" ) { A.qkv = L; A.have |= 1; }
                    else if ( leaf == "proj" ) { A.proj = L; A.have |= 2; }
                    else if ( leaf == "fc1" ) { A.fc1 = L; A.have |= 4; }
                    else if ( leaf == "fc2" ) { A.fc2 = L; A.have |= 8; }
                    else
                    {
                        error = "unexpected linear layer " + name;
                        return false;
                    }
                }
                else
                {
                    error = "unexpected linear layer " + name;
                    return false;
                }
                continue;
            }
            if ( name == "scalars.0" ) { scalars0_ = L; haveS0 = true; }
            else if ( name == "scalars.2" ) { scalars2_ = L; haveS2 = true; }
            else if ( name == "trunk.0" ) { trunk0_ = L; haveT0 = true; }
            else if ( name == "trunk.2" ) { trunk2_ = L; haveT2 = true; }
            else if ( name == "pi" ) { pi_ = L; havePi = true; }
            else if ( name == "v" ) { v_ = L; haveV = true; }
            else if ( name == "aux" ) { aux_ = L; hasAux_ = true; }
            else if ( name == "conditioning.0" ) { conditioning_[0] = L; haveConditioning[0] = true; }
            else if ( name == "conditioning.1" ) { conditioning_[1] = L; haveConditioning[1] = true; }
            else if ( name == "opponents.0" ) { opponents0_ = L; haveOpponents0 = true; }
            else if ( name == "opponents.2" ) { opponents2_ = L; haveOpponents2 = true; }
            else if ( name == "opponent_query" ) { opponentQuery_ = L; haveOpponentQuery = true; }
            else if ( name == "memory_projection" ) { memoryProjection_ = L; haveMemoryProjection = true; }
            else
            {
                error = "unexpected linear layer " + name;
                return false;
            }
        }
        else if ( kind == 3 )
        {
            LayerNorm N;
            uint32_t n;
            if ( !ReadU32( in, n ) || n == 0 || n > 4096 )
            {
                error = "truncated layer norm " + name;
                return false;
            }
            N.n = int( n );
            if ( !ReadFloats( in, dtype, n, N.w ) || !ReadFloats( in, dtype, n, N.b ) )
            {
                error = "truncated layer norm weights " + name;
                return false;
            }
            int tower, idx;
            std::string rest, leaf;
            if ( !TowerName( name, tower, rest ) || tower < 0 || tower >= int( towers_.size() ) ||
                 !IndexedName( rest, "attn.", idx, leaf ) || idx < 0 || idx >= 16 || ( leaf != "ln1" && leaf != "ln2" ) )
            {
                error = "unexpected layer norm " + name;
                return false;
            }
            Tower & tw = towers_[tower];
            if ( int( tw.attn.size() ) <= idx )
                tw.attn.resize( idx + 1 );
            if ( leaf == "ln1" ) { tw.attn[idx].ln1 = N; tw.attn[idx].have |= 16; }
            else { tw.attn[idx].ln2 = N; tw.attn[idx].have |= 32; }
        }
        else if ( kind == 4 )
        {
            uint32_t ndim;
            if ( !ReadU32( in, ndim ) || ndim == 0 || ndim > 4 )
            {
                error = "truncated parameter " + name;
                return false;
            }
            size_t count = 1;
            std::vector< uint32_t > dims;
            for ( uint32_t d = 0; d < ndim; ++d )
            {
                uint32_t dim;
                if ( !ReadU32( in, dim ) || dim == 0 || count > ( 1u << 24 ) / dim )
                {
                    error = "truncated parameter " + name;
                    return false;
                }
                count *= dim;
                dims.push_back( dim );
            }
            std::vector< float > values;
            if ( count > ( 1u << 24 ) || !ReadFloats( in, dtype, count, values ) )
            {
                error = "truncated parameter values " + name;
                return false;
            }
            if ( name == "memory_version" )
            {
                if ( dims != std::vector< uint32_t >{ 1 } || values[0] != 1.0f )
                {
                    error = "unsupported recurrent memory version";
                    return false;
                }
                haveMemoryVersion = true;
                continue;
            }
            int tower;
            std::string rest;
            if ( !TowerName( name, tower, rest ) || tower < 0 || tower >= int( towers_.size() ) || rest != "pos" ||
                 dims != std::vector< uint32_t >{ 1, 256, 128 } )
            {
                error = "unexpected parameter " + name;
                return false;
            }
            towers_[tower].pos.swap( values );
        }
        else if ( kind == 5 )
        {
            uint32_t inputs, hidden;
            if ( name != "memory_gru" || !ReadU32( in, inputs ) || !ReadU32( in, hidden ) ||
                 inputs != 775 || hidden != 256 )
            {
                error = "unsupported GRU dimensions or layer name";
                return false;
            }
            memoryInput_.in = int( inputs );
            memoryInput_.out = int( 3 * hidden );
            memoryHidden_.in = int( hidden );
            memoryHidden_.out = int( 3 * hidden );
            if ( !ReadFloats( in, dtype, size_t( 3 * hidden ) * inputs, memoryInput_.w ) ||
                 !ReadFloats( in, dtype, size_t( 3 * hidden ) * hidden, memoryHidden_.w ) ||
                 !ReadFloats( in, dtype, 3 * hidden, memoryInput_.b ) ||
                 !ReadFloats( in, dtype, 3 * hidden, memoryHidden_.b ) )
            {
                error = "truncated GRU weights";
                return false;
            }
            haveMemory = true;
        }
        else
        {
            error = "unknown layer kind";
            return false;
        }
    }
    for ( size_t t = 0; t < towers_.size(); ++t )
    {
        if ( !ValidV4Tower( towers_[t], t < 2 ) )
        {
            error = "v4 policy tower is incomplete or has incompatible dimensions";
            return false;
        }
    }
    if ( !haveS0 || !haveS2 || !haveT0 || !haveT2 || !havePi || !haveV )
    {
        error = "policy file is missing layers";
        return false;
    }
    if ( !haveMemoryVersion || !haveMemory || !haveMemoryProjection ||
         !LinearShape( memoryInput_, 775, 768 ) || !LinearShape( memoryHidden_, 256, 768 ) ||
         !LinearShape( memoryProjection_, 256, 768 ) )
    {
        error = "V4 memory schema 1 is required; recurrent layers are missing or incompatible";
        return false;
    }
    if ( !haveConditioning[0] || !haveConditioning[1] || !haveOpponents0 || !haveOpponents2 || !haveOpponentQuery ||
         !LinearShape( conditioning_[0], 256, 128 ) || !LinearShape( conditioning_[1], 256, 128 ) ||
         !LinearShape( opponents0_, 12, 64 ) || !LinearShape( opponents2_, 64, 64 ) ||
         !LinearShape( opponentQuery_, 896, 64 ) || !LinearShape( scalars0_, 72, 256 ) ||
         !LinearShape( scalars2_, 256, 256 ) || !LinearShape( trunk0_, 960, 768 ) ||
         !LinearShape( trunk2_, 768, 768 ) || !LinearShape( pi_, 768, 6 ) || !LinearShape( v_, 768, 1 ) ||
         !hasAux_ || !LinearShape( aux_, 768, 3 ) )
    {
        error = "v4 policy layers are incomplete or have incompatible dimensions";
        return false;
    }
    int fused = scalars2_.out + 64;
    for ( size_t t = 0; t < towers_.size(); ++t )
        fused += towers_[t].fc.out;
    if ( pi_.out != nActions_ || scalars0_.in != 72 || trunk0_.in != fused )
    {
        error = "policy file layers do not fit together";
        return false;
    }
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

void gNeuralNet::Net::RunTowerFeatures( Tower const & t, std::vector< float > & x, int & h, int & w ) const
{
    static std::vector< float > y, z, pooled, bias;
    int ho, wo;
    RunConvRaw( t.stem, x, h, w, y, ho, wo );
    ReluInPlace( y );
    x.swap( y );
    h = ho;
    w = wo;
    if ( t.haveReduce )
    {
        // Keep a residual block at 32x32 before reducing the local view to 16x16.
        RunConvRaw( t.pre.c0, x, h, w, y, ho, wo );
        ReluInPlace( y );
        RunConvRaw( t.pre.c1, y, ho, wo, z, h, w );
        for ( size_t i = 0; i < z.size(); ++i )
            z[i] = Relu( z[i] + x[i] );
        RunConvRaw( t.reduce, z, h, w, y, ho, wo );
        ReluInPlace( y );
        x.swap( y );
        h = ho;
        w = wo;
    }
    int const N = h * w;
    for ( size_t b = 0; b < t.blocks.size(); ++b )
    {
        Block const & B = t.blocks[b];
        RunConvRaw( B.c0, x, h, w, y, ho, wo );
        ReluInPlace( y );
        if ( B.gpool )
        {
            // the board-wide mean and max of the first GCH channels, fed back as a per-channel bias
            pooled.assign( size_t( 2 * B.gch ), 0.0f );
            for ( int c = 0; c < B.gch; ++c )
            {
                float const * plane = &y[size_t( c ) * N];
                float sum = 0, best = -1e30f;
                for ( int i = 0; i < N; ++i )
                {
                    sum += plane[i];
                    best = std::max( best, plane[i] );
                }
                pooled[c] = sum / N;
                pooled[B.gch + c] = best;
            }
            RunLinear( B.gp, &pooled[0], bias );
            for ( int c = 0; c < B.c0.cout; ++c )
            {
                float * plane = &y[size_t( c ) * N];
                for ( int i = 0; i < N; ++i )
                    plane[i] += bias[c];
            }
        }
        RunConvRaw( B.c1, y, h, w, z, ho, wo );
        for ( size_t i = 0; i < z.size(); ++i )
            z[i] += x[i];
        ReluInPlace( z );
        x.swap( z );
    }
    if ( !t.attn.empty() )
    {
        int const C = t.stem.cout;
        // the position embedding is stored [cell][channel]
        for ( int c = 0; c < C; ++c )
            for ( int i = 0; i < N; ++i )
                x[size_t( c ) * N + i] += t.pos[size_t( i ) * C + c];
        for ( size_t a = 0; a < t.attn.size(); ++a )
            RunAttention( t.attn[a], x, C, N );
    }
}

void gNeuralNet::Net::RunTowerHead( Tower const & t, std::vector< float > const & x, int h, int w, std::vector< float > & out ) const
{
    static std::vector< float > y, z;
    int ho, wo;
    RunConvRaw( t.down0, x, h, w, y, ho, wo );
    ReluInPlace( y );
    RunConvRaw( t.down1, y, ho, wo, z, h, w );
    ReluInPlace( z );
    RunLinear( t.fc, &z[0], out );
    ReluInPlace( out );
}

void gNeuralNet::Net::TowerInput( Tower const & t, uint8_t const * const * maps, Memory const & memory,
                                 std::vector< float > & x ) const
{
    // Each tower receives its current maps' planes in order, then the same previous maps.
    size_t cur = 0;
    for ( size_t j = 0; j < t.maps.size(); ++j )
        cur += mapPlanes_[t.maps[j]];
    size_t total = 2 * cur;
    x.clear();
    size_t off = 0;
    for ( size_t j = 0; j < t.maps.size(); ++j )
    {
        int m = t.maps[j];
        Unpack( maps[m], mapPlanes_[m], x, off, total );
        off += mapPlanes_[m];
    }
    for ( size_t j = 0; j < t.maps.size(); ++j )
    {
        int m = t.maps[j];
        Unpack( memory.have ? &memory.maps[m][0] : maps[m], mapPlanes_[m], x, off, total );
        off += mapPlanes_[m];
    }
}

int gNeuralNet::Net::Decide( uint8_t const * const * maps, float const * scalars, unsigned mask,
                             bool sample, float * probs, float * value, uint32_t & rng, Memory & memory,
                             float const * memoryContext )
{
    if ( memory.maps.size() != size_t( nMaps_ ) )
    {
        memory.maps.assign( nMaps_, std::vector< uint8_t >( size_t( grid_ ) * grid_, 0 ) );
        memory.have = false;
    }
    std::vector< float > h, part, t, x;
    std::vector< std::vector< float > > summaries( towers_.size() );
    for ( size_t i = 0; i < towers_.size(); ++i )
    {
        // First compute the arena summary that conditions both local views.
        size_t ti = ( i + 2 ) % 3;
        Tower const & tw = towers_[ti];
        TowerInput( tw, maps, memory, x );
        int height = grid_, width = grid_;
        RunTowerFeatures( tw, x, height, width );
        if ( ti < 2 )
        {
            RunLinear( conditioning_[ti], &summaries[2][0], part );
            int const cells = height * width, channels = tw.reduce.cout;
            for ( int c = 0; c < channels; ++c )
            {
                float const gain = 1.0f + 0.1f * std::tanh( part[c] );
                float const bias = 0.1f * std::tanh( part[channels + c] );
                for ( int cell = 0; cell < cells; ++cell )
                {
                    size_t at = size_t( c ) * cells + cell;
                    x[at] = Relu( x[at] * gain + bias );
                }
            }
        }
        RunTowerHead( tw, x, height, width, summaries[ti] );
    }
    for ( size_t i = 0; i < summaries.size(); ++i )
        h.insert( h.end(), summaries[i].begin(), summaries[i].end() );
    for ( int m = 0; m < nMaps_; ++m )
        memcpy( &memory.maps[m][0], maps[m], size_t( grid_ ) * grid_ );
    memory.have = true;
    // Enemy slots go exclusively through the shared opponent encoder below. Keeping them
    // out of this branch preserves slot-permutation invariance and masks absent enemies.
    std::vector< float > ownScalars( scalars, scalars + 64 );
    ownScalars.insert( ownScalars.end(), scalars + 88, scalars + 96 );
    RunLinear( scalars0_, &ownScalars[0], t );
    ReluInPlace( t );
    RunLinear( scalars2_, &t[0], part );
    ReluInPlace( part );
    h.insert( h.end(), part.begin(), part.end() );

    // Attention over the three observed opponents. Empty slots contribute exactly zero,
    // including the case where no opponents are currently present.
    std::vector< float > query, token[3], context( 64, 0.0f );
    RunLinear( opponentQuery_, &h[0], query );
    for ( size_t c = 0; c < query.size(); ++c )
        query[c] = std::tanh( query[c] );
    float scores[3], weights[3], best = -1e30f, sum = 0;
    for ( int o = 0; o < 3; ++o )
    {
        float opponent[12];
        std::copy( scalars + 64 + 8 * o, scalars + 64 + 8 * ( o + 1 ), opponent );
        std::copy( scalars + 96 + 4 * o, scalars + 96 + 4 * ( o + 1 ), opponent + 8 );
        RunLinear( opponents0_, opponent, t );
        ReluInPlace( t );
        RunLinear( opponents2_, &t[0], token[o] );
        ReluInPlace( token[o] );
        float dot = 0;
        for ( int c = 0; c < 64; ++c )
            dot += token[o][c] * query[c];
        scores[o] = opponent[7] > 0.5f ? dot / 8.0f : -1e4f;
        best = std::max( best, scores[o] );
    }
    for ( int o = 0; o < 3; ++o )
    {
        weights[o] = std::exp( scores[o] - best );
        sum += weights[o];
    }
    float presentSum = 0;
    for ( int o = 0; o < 3; ++o )
    {
        weights[o] = scalars[71 + 8 * o] > 0.5f ? weights[o] / sum : 0.0f;
        presentSum += weights[o];
    }
    for ( int o = 0; o < 3; ++o )
    {
        float weight = weights[o] / std::max( presentSum, 1e-9f );
        for ( int c = 0; c < 64; ++c )
            context[c] += weight * token[o][c];
    }
    h.insert( h.end(), context.begin(), context.end() );

    RunLinear( trunk0_, &h[0], t );
    ReluInPlace( t );
    RunLinear( trunk2_, &t[0], h );
    ReluInPlace( h );

    // PyTorch GRU order is reset, update, candidate. The reset gate multiplies
    // the entire hidden candidate affine term, including its bias.
    int const memorySize = 256;
    if ( memory.recurrent.size() != size_t( memorySize ) )
        memory.recurrent.assign( memorySize, 0.0f );
    std::vector< float > inputGates, hiddenGates, memoryInputs( h );
    for ( int c = 0; c < 7; ++c ) memoryInputs.push_back( memoryContext ? memoryContext[c] : 0.0f );
    RunLinear( memoryInput_, &memoryInputs[0], inputGates );
    RunLinear( memoryHidden_, &memory.recurrent[0], hiddenGates );
    for ( int i = 0; i < memorySize; ++i )
    {
        float const reset = 1.0f / ( 1.0f + std::exp( -( inputGates[i] + hiddenGates[i] ) ) );
        float const update = 1.0f / ( 1.0f + std::exp( -( inputGates[memorySize + i] + hiddenGates[memorySize + i] ) ) );
        float const candidate = std::tanh( inputGates[2 * memorySize + i] + reset * hiddenGates[2 * memorySize + i] );
        memory.recurrent[i] = ( 1.0f - update ) * candidate + update * memory.recurrent[i];
    }
    RunLinear( memoryProjection_, &memory.recurrent[0], part );
    for ( size_t i = 0; i < h.size(); ++i )
        h[i] += part[i];

    std::vector< float > logits, val;
    RunLinear( pi_, &h[0], logits );
    RunLinear( v_, &h[0], val );
    if ( value )
        *value = std::tanh( val[0] );

    best = -1e30f;
    for ( int a = 0; a < nActions_; ++a )
    {
        if ( !( ( mask >> a ) & 1 ) )
            logits[a] = -1e8f;
        best = std::max( best, logits[a] );
    }
    sum = 0;
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
