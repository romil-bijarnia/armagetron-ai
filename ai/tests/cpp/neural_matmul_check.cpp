// Exercise both the portable and selected backend with the same rectangular operands.
// Including the implementation exposes its internal helpers without adding a public game API.
#include "../../../src/tron/gNeuralNet.cpp"

#include <iostream>

int main()
{
    int32_t shape[5]; // M, N, K, transpose A, transpose B
    if ( !std::cin.read( reinterpret_cast< char * >( shape ), sizeof( shape ) ) ||
         shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 )
        return 2;
    int const M = shape[0], N = shape[1], K = shape[2];
    std::vector< float > A( size_t( M ) * K ), B( size_t( K ) * N ), C( size_t( M ) * N );
    if ( !std::cin.read( reinterpret_cast< char * >( &A[0] ), A.size() * sizeof( float ) ) ||
         !std::cin.read( reinterpret_cast< char * >( &B[0] ), B.size() * sizeof( float ) ) )
        return 2;
    MatmulPortable( &A[0], &B[0], &C[0], M, N, K, shape[3], shape[4] );
    std::cout.write( reinterpret_cast< char const * >( &C[0] ), C.size() * sizeof( float ) );
    Matmul( &A[0], &B[0], &C[0], M, N, K, shape[3], shape[4] );
    std::cout.write( reinterpret_cast< char const * >( &C[0] ), C.size() * sizeof( float ) );
    return std::cout ? 0 : 2;
}
