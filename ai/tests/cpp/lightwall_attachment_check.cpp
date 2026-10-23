#include "../../../src/tron/gLightwallAttachment.h"
#include "../../../src/tron/gLightcycleMotion.h"
#include <cassert>
#include <vector>

using namespace gLightwallAttachment;
void near(Point a,Point b) { assert(std::hypot(a.x-b.x,a.y-b.y)<1e-7); }

int main()
{
    // Straight runs attach at the fender, without pulling any settled vertices.
    auto straight=Curve::Join({0,0},{1,0},{1.4,0},{1,0},4);
    near(straight.Sample(0),{0,0});
    near(straight.Sample(1),{1.4,0});
    double previous=-1;
    for (int i=0;i<=100;++i)
    {
        auto p=straight.Sample(i/100.);
        assert(p.x>=previous && p.x<=1.4 && std::abs(p.y)<1e-10);
        previous=p.x;
    }
    // Moving the cut across a 90-degree corner cannot jump a control point.
    for (int sign : {-1,1})
    {
        auto before=Curve::Join({-1e-7,0},{1,0},{1,sign*2.},{0,double(sign)},1e-7);
        auto after=Curve::Join({0,sign*1e-7},{0,double(sign)},{1,sign*2.},{0,double(sign)},1e-7);
        for (int i=0;i<=100;++i)
        {
            auto a=before.Sample(i/100.),b=after.Sample(i/100.);
            assert(std::hypot(a.x-b.x,a.y-b.y)<1e-6);
        }
    }

    // A reversing emitter can cross the fixed wall's centreline. Changing the
    // preferred side must not flip the first handle across the wheel in one frame.
    for (double ahead : {0.,.3,1.0,1.8})
    {
        auto left=Curve::Join({ahead,-1e-7},{1,0},{0,0},{1,0},.2);
        auto right=Curve::Join({ahead,1e-7},{1,0},{0,0},{1,0},.2);
        for (int i=0;i<=100;++i)
        {
            auto a=left.Sample(i/100.),b=right.Sample(i/100.);
            assert(std::hypot(a.x-b.x,a.y-b.y)<1e-5);
        }
    }

    // Captured from the real SDL local-player stress run at 6.7-7.2 seconds.
    // The former curve had the correct endpoint/tangent but cut through the
    // rear tyre before reaching it. Coordinates are in the emitter's frame.
    struct WheelCase { Point start,incoming; double room; };
    WheelCase wheelCases[]={
        {{.0222586,.304806},{.889632,.456678},.107},
        {{.755564,.397928},{-.835956,-.548796},.299},
        {{1.052287,.553936},{-.911463,-.411382},.363},
        {{1.253438,.430313},{-.968757,-.248013},.382},
        {{1.430537,.463669},{-.998320,-.057944},.330},
        {{1.549287,.416934},{-.993158,.116775},.387}
    };
    for (auto const & test:wheelCases)
    for (int side:{-1,1})
    for (double scale:{1.,1.65/1.4})
    {
        auto curve=Curve::Join({test.start.x*scale,side*test.start.y*scale},
            {test.incoming.x,side*test.incoming.y},{0,0},{1,0},test.room*scale,.36*scale);
        near(curve.Sample(0),{test.start.x*scale,side*test.start.y*scale});
        near(curve.Sample(1),{0,0});
        for (int i=0;i<=200;++i)
        {
            Point p=curve.Sample(i/200.);
            assert(!(p.x>.056*scale && p.x<1.1*scale && std::abs(p.y)<.33*scale));
        }
    }

    // Exercise the actual visual pose through isolated turns, rapid alternating
    // turns and successive same-direction turns, at different renderer rates.
    for (int fps : {30,60,120,240})
    for (int mode=0;mode<3;++mode)
    {
        struct Segment { double begin,end; Point start,dir; };
        std::vector<Segment> path;
        gLightcycleMotion::State pose;
        Point position{0,0},direction{1,0};
        double distance=0,yaw=0,lastTurn=0;
        constexpr double rear=3.09375,speed=20; // authored bike at LIGHTCYCLE_SCALE 1.65
        path.push_back({0,0,position,direction});
        pose.Update(0,yaw,0,speed);
        double nextTurn=1;
        int turns=0;
        for (int frame=1;frame<fps*3;++frame)
        {
            double now=double(frame)/fps,then=double(frame-1)/fps;
            if (now>=nextTurn && (mode || turns==0))
            {
                double travel=(nextTurn-then)*speed;
                position=position+direction*travel;distance+=travel;
                path.back().end=distance;
                yaw+=gLightcycleMotion::Pi*.5*(mode==1 && turns%2 ? -1 : 1);
                direction={std::cos(yaw),std::sin(yaw)};
                lastTurn=nextTurn;nextTurn+=.075;++turns;
                path.push_back({distance,distance,position,direction});
                then=lastTurn;
            }
            position=position+direction*((now-then)*speed);
            distance+=(now-then)*speed;
            path.back().end=distance;
            pose.Update(now,yaw,lastTurn,speed);
            Point forward{std::cos(pose.Yaw()),std::sin(pose.Yaw())};
            Point emitter=position-forward*rear+Point{pose.VisualOffsetX(),pose.VisualOffsetY()};
            double cut=distance-Reach(rear);
            for (auto const & segment:path)
            {
                if (cut<segment.begin || cut>=segment.end) continue;
                Point start=segment.start+segment.dir*(cut-segment.begin);
                auto curve=Curve::Join(start,segment.dir,emitter,forward,
                    std::min(cut-segment.begin,segment.end-cut));
                near(curve.Sample(0),start);near(curve.Sample(1),emitter);
                auto tangent=curve.Tangent(1);
                assert(tangent.x*forward.x+tangent.y*forward.y>=0);
                assert(std::abs(tangent.x*forward.y-tangent.y*forward.x)<1e-7);
                auto approach=curve.Sample(.999)-emitter;
                assert(approach.x*forward.x+approach.y*forward.y<=1e-8);
                for (int i=0;i<=100;++i)
                {
                    auto p=curve.Sample(i/100.);
                    assert(std::isfinite(p.x) && std::isfinite(p.y));
                    assert(std::hypot(p.x-position.x,p.y-position.y)<Reach(rear)+2);
                    assert(curve.SpillWidth(i/100.)>=0 && curve.SpillWidth(i/100.)<=1);
                }
            }
        }
    }
}
