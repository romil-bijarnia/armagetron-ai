// Geometry for the short visual connection between a fixed wall and its emitter.
// Licensed under the project's GNU GPL version 2 or later.
#ifndef ARMAGETRON_LIGHTWALL_ATTACHMENT_H
#define ARMAGETRON_LIGHTWALL_ATTACHMENT_H

#include <algorithm>
#include <cmath>

namespace gLightwallAttachment
{
struct Point
{
    double x, y;
    Point operator+(Point b) const { return {x+b.x,y+b.y}; }
    Point operator-(Point b) const { return {x-b.x,y-b.y}; }
    Point operator*(double s) const { return {x*s,y*s}; }
};

struct Curve
{
    Point start, first, last, end;

    Point Sample(double t) const
    {
        t=std::max(0.,std::min(1.,t));
        double s=1-t;
        return start*(s*s*s)+first*(3*s*s*t)+last*(3*s*t*t)+end*(t*t*t);
    }

    Point Tangent(double t) const
    {
        double s=1-t;
        return (first-start)*(3*s*s)+(last-first)*(6*s*t)+(end-last)*(3*t*t);
    }

    double SpillWidth(double t) const
    {
        Point tangent=Tangent(t);
        Point bend=(last-first*2+start)*(6*(1-t))+(end-last*2+first)*(6*t);
        double speed=std::hypot(tangent.x,tangent.y);
        double cross=std::abs(tangent.x*bend.y-tangent.y*bend.x);
        // Offset ribbons fold over themselves if wider than the local bend.
        return cross>1e-9 ? std::min(1.,.55*speed*speed*speed/cross) : 1.;
    }

    static Curve Join(Point start,Point incoming,Point end,Point forward,double cornerRoom)
    {
        double span=std::hypot(end.x-start.x,end.y-start.y);
        // The start handle vanishes at both sides of a physical corner. As
        // the cut crosses it, the curve remains continuous despite the 90deg
        // change of the wall tangent. The emitter always has a rearward handle.
        double firstLength=std::min(span*.35,std::max(0.,cornerRoom));
        double lastLength=std::min(1.2,span*.55);
        return {start,start+incoming*firstLength,end-forward*lastLength,end};
    }
};

// Extra distance behind the rear fender where the fixed wall becomes a curve.
// It stays inside the existing five-metre uncached wall window at normal scale.
inline double Reach(double rearDistance) { return std::min(4.9,rearDistance+1.4); }
}
#endif
