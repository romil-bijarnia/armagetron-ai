#include "../../../src/tron/gImpactAnimation.h"
#include <cassert>
#include <cmath>
#include <limits>
int main()
{
    using namespace gImpactAnimation;
    for (unsigned i=0;i<FragmentCount;++i)
        for (int frame=0;frame<=240;++frame)
        {
            float t=frame/120.f;
            auto a=Sample(i,t,20),b=Sample(i,t,20),fast=Sample(i,t,60);
            assert(a.along==b.along && a.across==b.across && a.height==b.height);
            assert(a.spin==b.spin && a.fade==b.fade); // repeated reflection/main sample
            assert(std::isfinite(a.along) && std::isfinite(a.height));
            if (!a.Active()) continue;
            assert(a.height>=.025f && a.height<4);
            assert(a.fade<=1 && a.sizeX>0 && a.sizeY>0 && a.sizeZ>0);
            assert(a.sizeZ<a.sizeX); // flat shards, not uniform cubes
            assert(fast.along>=a.along); // stronger incoming momentum
            assert(fast.along-a.along<=3.5f); // impulse is bounded by drag
            assert(std::abs(a.across)<3);
        }
    assert(Sample(0,0,20).Active()); // initial vehicle panels exist at contact
    assert(!Sample(80,0,20).Active()); // smaller debris follows the contact flash
    assert(Sample(80,.15f,20).Active());
    assert(!Sample(0,Duration,20).Active());
    assert(!Sample(0,-.1f,20).Active());
    assert(!Sample(0,std::numeric_limits<float>::quiet_NaN(),20).Active());
    assert(!Sample(0,.2f,std::numeric_limits<float>::infinity()).Active());
    assert(!Sample(180,0,20).Active()); // fine particles follow the shell breakup
    assert(Sample(180,.20f,20).Active());
    for (unsigned i=0;i<RimFragmentCount;++i)
    {
        auto initial=SampleRim(i,0,20,.7f);
        assert(initial.Active() && initial.pitch==0 && initial.roll==0);
        // The authored long-shell bike's rear axle is at -1.45.
        float axle=((i/12>=2 ? 1.55f : -1.45f)-1.5f)*.7f;
        float x=initial.along-axle,z=initial.height-.72f*.7f;
        assert(std::abs(std::hypot(x,z)-.607f*.7f)<.0001f);
        assert(std::abs(std::abs(initial.across)-.446f*.7f)<.0001f);
        for (int frame=0;frame<240;++frame)
        {
            float t=frame/120.f;
            auto a=SampleRim(i,t,20,.7f),b=SampleRim(i,t,20,.7f),fast=SampleRim(i,t,60,.7f);
            assert(a.along==b.along && a.height==b.height && a.pitch==b.pitch && a.roll==b.roll);
            assert(std::isfinite(a.along) && std::isfinite(a.height));
            if (!a.Active()) continue;
            assert(a.height>=.035f && a.height<3 && a.fade<=1);
            assert(fast.along>=a.along && fast.along-a.along<=2.2f);
        }
    }
    assert(!SampleRim(RimFragmentCount,.2f,20,.7f).Active());
    assert(!SampleRim(0,.2f,20,0).Active());
    assert(!SampleRim(0,-.01f,20,.7f).Active());
    assert(!SampleRim(0,.2f,20,std::numeric_limits<float>::infinity()).Active());
}
