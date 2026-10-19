// Opponent observation layout 2: appended resources/timers preserve every original input.
#include "../../sim/tron.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace
{
struct Observation
{
    std::array< uint8_t, 4 * 64 * 64 > maps{};
    std::array< float, tron::World::kScalars > scalars{};
};

Observation Observe( tron::World const & world )
{
    Observation out;
    std::fill( out.scalars.begin(), out.scalars.end(), -99.f );
    world.Observe( 0, out.maps.data(), out.maps.data() + 4096, out.maps.data() + 8192,
                   out.maps.data() + 12288, out.scalars.data() );
    assert( std::all_of( out.scalars.begin() + 88, out.scalars.begin() + 96,
                        []( float x ) { return x == 0; } ) );
    return out;
}

void SameOriginalInputs( Observation const & a, Observation const & b )
{
    assert( a.maps == b.maps );
    assert( std::equal( a.scalars.begin(), a.scalars.begin() + 96, b.scalars.begin() ) );
}

bool Near( float a, float b ) { return std::fabs( a - b ) < 1e-5f; }
}

int main()
{
    static_assert( tron::World::kScalars == 108, "observation layout 2 has 108 scalars" );
    tron::World full;
    full.Reset( -3, 2 );
    full.Place( 0, 40, 80, 0, 1 );
    full.Place( 1, 100, 80, 1, 0 );
    tron::World empty = full;
    empty.cycles[1].brakingReservoir = 0;
    Observation a = Observe( full ), b = Observe( empty );
    SameOriginalInputs( a, b );
    assert( a.scalars[96] == 1 && b.scalars[96] == 0 );
    assert( std::all_of( a.scalars.begin() + 100, a.scalars.end(), []( float x ) { return x == 0; } ) );
    int actions[2] = { tron::ACT_STRAIGHT, tron::ACT_BRAKE };
    for ( int i = 0; i < 10; ++i )
    {
        full.Step( actions );
        empty.Step( actions );
    }
    assert( full.cycles[1].Speed() < empty.cycles[1].Speed() );

    // Rubber and turn timers were also invisible in the original enemy token.
    tron::World resources;
    resources.Reset( -3, 2 );
    resources.Place( 0, 40, 80, 0, 1 );
    resources.Place( 1, 100, 80, 1, 0 );
    resources.time = 1.5f;
    resources.cycles[1].lastTime = 1;
    a = Observe( resources );
    resources.cycles[1].rubber = .7f * resources.s.rubber;
    b = Observe( resources );
    SameOriginalInputs( a, b );
    assert( Near( b.scalars[97], .7f ) );
    float const delay = resources.s.delay * resources.s.delayBonus;
    for ( float wait : { 0.f, .5f, 1.f } )
    {
        resources.cycles[1].lastTurnLeft = 1 - ( 1 - wait ) * delay;
        Observation timed = Observe( resources );
        SameOriginalInputs( b, timed );
        assert( Near( timed.scalars[98], wait ) && Near( timed.scalars[99], wait ) );
    }
    resources.cycles[1].lastTurnLeft = 2;
    assert( Observe( resources ).scalars[98] == 1 );
    resources.cycles[1].lastTurnLeft = -10;
    resources.s.delay = resources.s.rubber = 0;
    b = Observe( resources );
    assert( b.scalars[97] == 0 && b.scalars[98] == 0 && b.scalars[99] == 0 );

    // The appended blocks follow the same nearest-enemy sort as the original 8-value tokens.
    tron::World sorted;
    sorted.Reset( -3, 4 );
    sorted.Place( 0, 80, 80, 0, 1 );
    sorted.Place( 1, 80, 120, 1, 0 );
    sorted.Place( 2, 80, 100, 1, 0 );
    sorted.Place( 3, 80, 110, 1, 0 );
    for ( int i = 1; i < 4; ++i )
    {
        sorted.cycles[i].brakingReservoir = .2f * i;
        sorted.cycles[i].rubber = .3f * i * sorted.s.rubber;
    }
    a = Observe( sorted );
    int const order[3] = { 2, 3, 1 };
    for ( int slot = 0; slot < 3; ++slot )
    {
        assert( Near( a.scalars[64 + 8 * slot], .2f + .1f * slot ) );
        assert( a.scalars[71 + 8 * slot] == 1 );
        assert( Near( a.scalars[96 + 4 * slot], .2f * order[slot] ) );
        assert( Near( a.scalars[97 + 4 * slot], .3f * order[slot] ) );
    }
    sorted.Reset( -3, 1 );
    a = Observe( sorted );
    assert( std::all_of( a.scalars.begin() + 64, a.scalars.end(), []( float x ) { return x == 0; } ) );
}
