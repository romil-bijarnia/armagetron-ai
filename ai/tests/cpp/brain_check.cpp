// Runs the game's C++ policy network on recorded inputs so the Python test can compare it with
// PyTorch. usage: brain_check policy.bin inputs.bin [reset_index]
// inputs.bin: u32 n, then per case: NumMaps() grids of grid*grid bytes, n_scalars float32, u8 mask.
// With fourth argument "memory", each case appends seven context floats and output appends state.
// Prints per case: action probs... value. Reset temporal memory before reset_index, if supplied.
#include "../../../src/tron/gNeuralNet.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

int main( int argc, char ** argv )
{
    if ( argc < 3 )
    {
        std::cerr << "usage: brain_check policy.bin inputs.bin [reset_index]\n";
        return 2;
    }
    gNeuralNet::Net net;
    std::ifstream f( argv[1], std::ios::binary );
    std::string error;
    if ( !net.Load( f, error ) )
    {
        std::cerr << "load failed: " << error << "\n";
        return 1;
    }
    std::ifstream in( argv[2], std::ios::binary );
    uint32_t n = 0;
    in.read( reinterpret_cast< char * >( &n ), 4 );
    int const g2 = net.Grid() * net.Grid();
    std::vector< std::vector< uint8_t > > maps( net.NumMaps(), std::vector< uint8_t >( g2 ) );
    std::vector< uint8_t const * > mapPtrs( net.NumMaps() );
    for ( int m = 0; m < net.NumMaps(); ++m )
        mapPtrs[m] = &maps[m][0];
    std::vector< float > scalars( net.NumScalars() ), probs( net.NumActions() );
    double total = 0;
    uint32_t rng = 1;
    int const resetIndex = argc > 3 ? std::stoi( argv[3] ) : -1;
    bool const withMemory = argc > 4 && std::string( argv[4] ) == "memory";
    gNeuralNet::Net::Memory memory; // the cases are one cycle's decisions in a row
    for ( uint32_t i = 0; i < n; ++i )
    {
        uint8_t mask;
        for ( int m = 0; m < net.NumMaps(); ++m )
            in.read( reinterpret_cast< char * >( &maps[m][0] ), g2 );
        in.read( reinterpret_cast< char * >( &scalars[0] ), scalars.size() * 4 );
        in.read( reinterpret_cast< char * >( &mask ), 1 );
        float context[7] = {};
        if ( withMemory ) in.read( reinterpret_cast< char * >( context ), sizeof( context ) );
        if ( !in )
        {
            std::cerr << "truncated input case " << i << "\n";
            return 2;
        }
        if ( int( i ) == resetIndex )
            memory.Reset();
        float value = 0;
        auto t0 = std::chrono::steady_clock::now();
        int a = net.Decide( &mapPtrs[0], &scalars[0], mask, false, &probs[0], &value, rng, memory, context );
        total += std::chrono::duration< double, std::milli >( std::chrono::steady_clock::now() - t0 ).count();
        std::printf( "%d", a );
        for ( int k = 0; k < net.NumActions(); ++k )
            std::printf( " %.9g", probs[k] );
        std::printf( " %.9g", value );
        if ( withMemory ) for ( float hidden : memory.recurrent ) std::printf( " %.9g", hidden );
        std::printf( "\n" );
    }
    std::fprintf( stderr, "update %d, %.2f ms per decision\n", net.Update(), n ? total / n : 0.0 );
    return 0;
}
