
#include "tron.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <iostream>
int main() {
    tron::World full;
    full.Reset(-3,2);
    full.Place(0,40,80,0,1);
    full.Place(1,100,80,1,0);
    tron::World empty=full;
    full.cycles[1].brakingReservoir=1;
    empty.cycles[1].brakingReservoir=0;
    std::array<uint8_t,4*64*64> mf{}, me{};
    std::array<float,96> sf{}, se{};
    full.Observe(0,mf.data(),mf.data()+4096,mf.data()+8192,mf.data()+12288,sf.data());
    empty.Observe(0,me.data(),me.data()+4096,me.data()+8192,me.data()+12288,se.data());
    std::cout << "identical_maps=" << (mf==me) << " identical_96_scalars=" << (sf==se)
              << " identical_legal_mask=" << (full.ActionMask(0)==empty.ActionMask(0)) << "\n";
    std::cout << "reserved_scalars_88_95_all_zero=" << std::all_of(sf.begin()+88,sf.end(),[](float x){return x==0;}) << "\n";
    int actions[2]={0,3};
    for(int i=0;i<10;++i) {full.Step(actions);empty.Step(actions);}
    std::cout << "after_identical_controls_seconds=" << full.time << " enemy_speed_full_reserve=" << full.cycles[1].Speed()
              << " enemy_speed_empty_reserve=" << empty.cycles[1].Speed() << "\n";
    return mf==me && sf==se && full.cycles[1].Speed()!=empty.cycles[1].Speed() ? 0 : 1;
}
