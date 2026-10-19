/*
 * ArmageTron -- Just another Tron Lightcycle Game in 3D.
 * Copyright (C) 2026, Armagetron Advanced Development Team
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2 or later.
 */

#ifndef ARMAGETRON_LIGHTCYCLE_MODEL_H
#define ARMAGETRON_LIGHTCYCLE_MODEL_H

#include "aa_config.h"

// Model space follows the original bike: forward is +X, the axles run along Y,
// and the contact plane is Z=0. The caller supplies heading, lean and world scale.
namespace gLightcycleModel
{
constexpr float RearAxle = -1.15f;
constexpr float FrontAxle = 1.55f;
constexpr float WheelRadius = .72f;
#ifdef DEDICATED
inline void Render(float, float, float, float, float) {}
#else
}

#include "rGL.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
#ifdef HAVE_GLEW
#include "rScreen.h"
#include <map>
#endif

namespace gLightcycleModel
{
namespace Detail
{
float const Pi = 3.14159265358979323846f;

struct Vec
{
    float x, y, z;
    Vec(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
    Vec operator +(Vec const & v) const { return Vec(x + v.x, y + v.y, z + v.z); }
    Vec operator -(Vec const & v) const { return Vec(x - v.x, y - v.y, z - v.z); }
    Vec operator *(float f) const { return Vec(x * f, y * f, z * f); }
};

inline float Dot(Vec const & a, Vec const & b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec Cross(Vec const & a, Vec const & b)
{
    return Vec(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline Vec Unit(Vec const & v)
{
    float length = std::sqrt(Dot(v, v));
    return length > .000001f ? v * (1.f / length) : Vec(0, 0, 1);
}
inline Vec Catmull(Vec const & a, Vec const & b, Vec const & c, Vec const & d, float t)
{
    return (b * 2.f + (c - a) * t
            + (a * 2.f - b * 5.f + c * 4.f - d) * (t*t)
            + (b * 3.f - a - c * 3.f + d) * (t*t*t)) * .5f;
}

struct Vertex
{
    Vec position, normal;
    Vertex(Vec const & p, Vec const & n) : position(p), normal(n) {}
};

#ifdef HAVE_GLEW
struct Mesh;
inline void ReleaseVertexBuffers();

// A zero entry records a failed allocation so an old/low-memory driver is not
// asked to upload the same mesh every frame. Reset the cache with the GL context.
inline std::map<Mesh const *, GLuint> & VertexBuffers()
{
    static std::map<Mesh const *, GLuint> buffers;
    static rCallbackBeforeScreenModeChange cleanup(&ReleaseVertexBuffers);
    (void)cleanup;
    return buffers;
}

inline void ReleaseVertexBuffers()
{
    std::map<Mesh const *, GLuint> & buffers = VertexBuffers();
    if (GLEW_VERSION_1_5)
        for (auto const & entry : buffers)
            if (entry.second)
                glDeleteBuffers(1, &entry.second);
    buffers.clear();
}

inline GLuint UploadVertexBuffer(std::vector<Vertex> const & vertices)
{
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    if (buffer)
    {
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        GLsizeiptr bytes = static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex));
        glBufferData(GL_ARRAY_BUFFER, bytes, vertices.data(), GL_STATIC_DRAW);
        GLint allocated = 0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &allocated);
        if (allocated == bytes)
            return buffer;
        glDeleteBuffers(1, &buffer);
    }
    // Allocation errors belong to this optional path. Leave a clean client-array
    // fallback rather than letting an out-of-memory flag disable a later effect.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    while (glGetError() != GL_NO_ERROR)
        ;
    return 0;
}
#endif

struct Mesh
{
    std::vector<Vertex> vertices;

    void Triangle(Vertex a, Vertex b, Vertex c)
    {
        Vec face = Cross(b.position - a.position, c.position - a.position);
        if (Dot(face, face) < 1.e-15f)
            return;
        if (Dot(face, a.normal + b.normal + c.normal) < 0)
            std::swap(b, c);
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
    }

    void Quad(Vertex const & a, Vertex const & b, Vertex const & c, Vertex const & d)
    {
        Triangle(a, b, c);
        Triangle(a, c, d);
    }

    void Draw() const
    {
        if (vertices.empty())
            return;
#ifdef HAVE_GLEW
        if (GLEW_VERSION_1_5)
        {
            std::map<Mesh const *, GLuint> & buffers = VertexBuffers();
            auto entry = buffers.find(this);
            if (entry == buffers.end())
                entry = buffers.emplace(this, UploadVertexBuffer(vertices)).first;
            glBindBuffer(GL_ARRAY_BUFFER, entry->second);
            if (entry->second)
            {
                glVertexPointer(3, GL_FLOAT, sizeof(Vertex), reinterpret_cast<void const *>(offsetof(Vertex, position)));
                glNormalPointer(GL_FLOAT, sizeof(Vertex), reinterpret_cast<void const *>(offsetof(Vertex, normal)));
                glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
                return;
            }
        }
#endif
        unsigned char const * first = reinterpret_cast<unsigned char const *>(vertices.data());
        glVertexPointer(3, GL_FLOAT, sizeof(Vertex), first + offsetof(Vertex, position));
        glNormalPointer(GL_FLOAT, sizeof(Vertex), first + offsetof(Vertex, normal));
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
    }
};

enum Material
{
    Shell, Armour, Metal, Rubber, Recess, Suit, Visor,
    Light, SubtleLight, Headlight, MaterialCount
};
typedef std::array<Mesh, MaterialCount> Parts;

template<class Function>
inline void Surface(Mesh & mesh, int columns, int rows, Function sample)
{
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < columns; ++x)
        {
            float u0 = float(x) / columns, u1 = float(x + 1) / columns;
            float v0 = float(y) / rows, v1 = float(y + 1) / rows;
            mesh.Quad(sample(u0, v0), sample(u1, v0), sample(u1, v1), sample(u0, v1));
        }
}

inline void Ellipsoid(Mesh & mesh, Vec const & center, Vec const & radius,
                      int longitude = 28, int latitude = 16)
{
    Surface(mesh, longitude, latitude, [&](float u, float v)
    {
        float a = 2.f * Pi * u, b = Pi * (v - .5f);
        Vec n(std::cos(b) * std::cos(a), std::cos(b) * std::sin(a), std::sin(b));
        Vec p(n.x * radius.x, n.y * radius.y, n.z * radius.z);
        return Vertex(center + p, Unit(Vec(n.x / radius.x, n.y / radius.y, n.z / radius.z)));
    });
}

inline void Rod(Mesh & mesh, Vec const & a, Vec const & b, float ra, float rb,
                int sides = 14)
{
    Vec axis = b - a;
    float length = std::sqrt(Dot(axis, axis));
    Vec direction = Unit(axis);
    Vec u = Unit(Cross(direction, std::fabs(direction.z) < .95f ? Vec(0, 0, 1) : Vec(0, 1, 0)));
    Vec v = Cross(direction, u);
    for (int i = 0; i < sides; ++i)
    {
        float p0 = 2 * Pi * i / sides, p1 = 2 * Pi * (i + 1) / sides;
        Vec radial0 = u * std::cos(p0) + v * std::sin(p0);
        Vec radial1 = u * std::cos(p1) + v * std::sin(p1);
        Vec n0 = Unit(radial0 + direction * ((ra - rb) / std::max(.0001f, length)));
        Vec n1 = Unit(radial1 + direction * ((ra - rb) / std::max(.0001f, length)));
        mesh.Quad(Vertex(a + radial0 * ra, n0), Vertex(b + radial0 * rb, n0),
                  Vertex(b + radial1 * rb, n1), Vertex(a + radial1 * ra, n1));
        mesh.Triangle(Vertex(a, direction * -1), Vertex(a + radial1 * ra, direction * -1),
                      Vertex(a + radial0 * ra, direction * -1));
        mesh.Triangle(Vertex(b, direction), Vertex(b + radial0 * rb, direction),
                      Vertex(b + radial1 * rb, direction));
    }
}

// A closed profile revolved around the axle. Normals follow the profile instead
// of flattening the broad tyre shoulders, metal lips or the curved fender crown.
struct Radial
{
    float y, radius;
};

inline void Lathe(Mesh & mesh, std::vector<Radial> const & profile, Vec const & center,
                  float start = 0, float finish = 2 * Pi, int segments = 72)
{
    int count = static_cast<int>(profile.size());
    for (int j = 0; j < count; ++j)
    {
        int next = (j + 1) % count;
        auto sample = [&](int k, float angle)
        {
            Radial const & previous = profile[(k + count - 1) % count];
            Radial const & following = profile[(k + 1) % count];
            float dy = following.y - previous.y, dr = following.radius - previous.radius;
            float c = std::cos(angle), s = std::sin(angle);
            return Vertex(center + Vec(profile[k].radius * c, profile[k].y, profile[k].radius * s),
                          Unit(Vec(dy * c, -dr, dy * s)));
        };
        for (int i = 0; i < segments; ++i)
        {
            float a = start + (finish - start) * i / segments;
            float b = start + (finish - start) * (i + 1) / segments;
            mesh.Quad(sample(j, a), sample(j, b), sample(next, b), sample(next, a));
        }
        if (finish - start < 2 * Pi - .001f)
        {
            Radial middle = {0, 0};
            for (Radial const & p : profile)
            {
                middle.y += p.y / count;
                middle.radius += p.radius / count;
            }
            for (int cap = 0; cap < 2; ++cap)
            {
                float angle = cap ? finish : start;
                Vec n(std::sin(angle), 0, -std::cos(angle));
                if (cap)
                    n = n * -1;
                Vec c = center + Vec(middle.radius * std::cos(angle), middle.y, middle.radius * std::sin(angle));
                Vertex a = sample(j, angle), b = sample(next, angle);
                mesh.Triangle(Vertex(c, n), Vertex(a.position, n), Vertex(b.position, n));
            }
        }
    }
}

inline void Annulus(Mesh & mesh, float inner, float outer, float y, Vec const & center,
                    float start = 0, float finish = 2 * Pi, int segments = 72)
{
    Vec n(0, y < 0 ? -1.f : 1.f, 0);
    for (int i = 0; i < segments; ++i)
    {
        float a = start + (finish - start) * i / segments;
        float b = start + (finish - start) * (i + 1) / segments;
        auto p = [&](float r, float t) { return Vertex(center + Vec(r * std::cos(t), y, r * std::sin(t)), n); };
        mesh.Quad(p(inner, a), p(outer, a), p(outer, b), p(inner, b));
    }
}

inline Vec PathPoint(std::vector<Vec> const & path, float t)
{
    float scaled = std::max(0.f, std::min(1.f, t)) * (path.size() - 1);
    int i = std::min(static_cast<int>(path.size()) - 2, static_cast<int>(scaled));
    return Catmull(path[std::max(0, i - 1)], path[i], path[i + 1],
                   path[std::min(static_cast<int>(path.size()) - 1, i + 2)], scaled - i);
}

inline void Tube(Mesh & mesh, std::vector<Vec> const & path, float radius, int sides = 8)
{
    Surface(mesh, static_cast<int>(path.size() - 1) * 7, sides, [&](float u, float v)
    {
        Vec p = PathPoint(path, u);
        Vec tangent = Unit(PathPoint(path, std::min(1.f, u + .001f))
                         - PathPoint(path, std::max(0.f, u - .001f)));
        Vec first = Unit(Cross(tangent, std::fabs(tangent.y) < .95f ? Vec(0, 1, 0) : Vec(0, 0, 1)));
        Vec second = Cross(tangent, first);
        Vec n = first * std::cos(2 * Pi * v) + second * std::sin(2 * Pi * v);
        return Vertex(p + n * radius, n);
    });
}

struct Section
{
    float x, z, width, height;
};

inline Section Interpolate(std::vector<Section> const & sections, float u)
{
    float scaled = std::max(0.f, std::min(1.f, u)) * (sections.size() - 1);
    int i = std::min(static_cast<int>(sections.size()) - 2, static_cast<int>(scaled));
    Section const & a = sections[std::max(0, i - 1)];
    Section const & b = sections[i];
    Section const & c = sections[i + 1];
    Section const & d = sections[std::min(static_cast<int>(sections.size()) - 1, i + 2)];
    float t = scaled - i;
    Vec geometry = Catmull(Vec(a.x, a.z, a.width), Vec(b.x, b.z, b.width),
                          Vec(c.x, c.z, c.width), Vec(d.x, d.z, d.width), t);
    Vec vertical = Catmull(Vec(a.height, 0, 0), Vec(b.height, 0, 0),
                          Vec(c.height, 0, 0), Vec(d.height, 0, 0), t);
    return Section{geometry.x, geometry.y, std::max(.003f, geometry.z), std::max(.003f, vertical.x)};
}

inline Vec LoftPoint(std::vector<Section> const & sections, float u, float angle)
{
    Section p = Interpolate(sections, u);
    return Vec(p.x, p.width * std::cos(angle), p.z + p.height * std::sin(angle));
}

inline void Loft(Mesh & mesh, std::vector<Section> const & sections)
{
    Surface(mesh, static_cast<int>(sections.size() - 1) * 7, 28, [&](float u, float v)
    {
        float angle = 2 * Pi * v;
        Vec along = LoftPoint(sections, std::min(1.f, u + .001f), angle)
                  - LoftPoint(sections, std::max(0.f, u - .001f), angle);
        Vec around = LoftPoint(sections, u, angle + .001f) - LoftPoint(sections, u, angle - .001f);
        return Vertex(LoftPoint(sections, u, angle), Unit(Cross(around, along)));
    });
    for (int end = 0; end < 2; ++end)
    {
        float u = float(end);
        Section p = Interpolate(sections, u);
        Vec n(end ? 1.f : -1.f, 0, 0);
        for (int i = 0; i < 28; ++i)
            mesh.Triangle(Vertex(Vec(p.x, 0, p.z), n),
                          Vertex(LoftPoint(sections, u, 2 * Pi * i / 28), n),
                          Vertex(LoftPoint(sections, u, 2 * Pi * (i + 1) / 28), n));
    }
}

inline void Stripe(Mesh & mesh, std::vector<Section> const & sections, float angle, float radius)
{
    std::vector<Vec> path;
    for (int i = 0; i <= 28; ++i)
        path.push_back(LoftPoint(sections, float(i) / 28, angle) + Vec(0, std::cos(angle) * .004f, std::sin(angle) * .004f));
    Tube(mesh, path, radius);
}

// Flush bands follow the fairing surface; they are inlaid panels, not glowing
// pipes floating above the paint.
inline void FairingBand(Mesh & mesh, std::vector<Section> const & sections,
                        float angle, float width, float offset=.004f)
{
    Surface(mesh, static_cast<int>(sections.size()-1)*10, 3, [&](float u,float v)
    {
        float a=angle+(v-.5f)*width;
        Vec along=LoftPoint(sections,std::min(1.f,u+.001f),a)-LoftPoint(sections,std::max(0.f,u-.001f),a);
        Vec around=LoftPoint(sections,u,a+.001f)-LoftPoint(sections,u,a-.001f);
        Vec n=Unit(Cross(around,along));
        return Vertex(LoftPoint(sections,u,a)+n*offset,n);
    });
}

inline void CrownBand(Mesh & mesh,Vec const & center,float y0,float y1,
                      float radius,float start,float finish)
{
    Surface(mesh,72,3,[&](float u,float v)
    {
        float a=start+(finish-start)*u;
        float y=y0+(y1-y0)*v;
        // Match the broad shroud's gently rounded crown.
        float r=radius-.035f*(y/.45f)*(y/.45f);
        Vec n=Unit(Vec(std::cos(a),.07f*y/(.45f*.45f),std::sin(a)));
        return Vertex(center+Vec(r*std::cos(a),y,r*std::sin(a)),n);
    });
}

struct Cache
{
    Parts body, wheel;

    Cache()
    {
        BuildWheel();
        BuildChassis();
        BuildRider();
    }

    void BuildWheel()
    {
        // A deep annular wheel: no central axle, spokes or fork. Most of the
        // visible side opening is empty and the silver barrel recedes into it.
        Lathe(wheel[Rubber],{{-.36f,.465f},{-.415f,.52f},{-.442f,.59f},
            {-.433f,.65f},{-.38f,.690f},{-.27f,.716f},{-.12f,.72f},
            {.12f,.72f},{.27f,.716f},{.38f,.690f},{.433f,.65f},
            {.442f,.59f},{.415f,.52f},{.36f,.465f}},Vec(),0,2*Pi,112);
        Lathe(wheel[Metal],{{-.405f,.478f},{-.386f,.499f},{-.34f,.487f},
            {-.24f,.451f},{-.12f,.423f},{.12f,.423f},{.24f,.451f},
            {.34f,.487f},{.386f,.499f},{.405f,.478f},
            {.23f,.435f},{.12f,.411f},{-.12f,.411f},{-.23f,.435f}},Vec(),0,2*Pi,112);
        for(int side=-1;side<=1;side+=2)
        {
            float s=float(side);
            Annulus(wheel[Shell],.492f,.568f,s*.443f,Vec(),0,2*Pi,112);
            Annulus(wheel[Light],.574f,.640f,s*.446f,Vec(),0,2*Pi,144);
            Annulus(wheel[SubtleLight],.565f,.573f,s*.447f,Vec(),0,2*Pi,144);
            Annulus(wheel[Armour],.642f,.661f,s*.435f,Vec(),0,2*Pi,112);
            Annulus(wheel[SubtleLight],.491f,.500f,s*.407f,Vec(),0,2*Pi,112);
            // Recessed register marks stay on the annular lip, never across the
            // opening. They give scale without reconstructing a spoked wheel.
            for(int i=0;i<32;++i)
            {
                float a=2*Pi*i/32;
                Annulus(wheel[Armour],.514f,.535f,s*.445f,Vec(),a,a+.018f,2);
            }
        }
        // Twin luminous tread inlays remain visible in the unchanged rear view.
        for(int side=-1;side<=1;side+=2)
            CrownBand(wheel[Light],Vec(),side*.12f-.013f,side*.12f+.013f,.723f,0,2*Pi);
        for(int i=0;i<40;++i)
        {
            float center=2*Pi*i/40;
            Surface(wheel[Recess],1,8,[&](float u,float v)
            {
                float y=-.26f+.52f*v;
                float angle=center+(u-.5f)*.007f+y*.19f;
                float radius=.722f-.009f*std::pow(std::fabs(y)/.27f,3.f);
                return Vertex(Vec(radius*std::cos(angle),y,radius*std::sin(angle)),
                              Unit(Vec(std::cos(angle),.02f*y,std::sin(angle))));
            });
        }
    }

    void BuildChassis()
    {
        // One long, low sculpted chassis joins the wheel housings. The lower
        // fairing sweeps under the reactor and rises into the broad front dome.
        std::vector<Section> belly={
            {-.66f,.43f,.37f,.07f},{-.48f,.34f,.395f,.12f},
            {-.27f,.29f,.405f,.13f},
            {.12f,.29f,.405f,.13f},{.43f,.43f,.40f,.21f},
            {.69f,.70f,.405f,.25f},{.93f,.98f,.41f,.24f},
            {1.16f,1.24f,.42f,.17f},{1.43f,1.39f,.34f,.075f},
            {1.68f,1.40f,.22f,.04f}
        };
        std::vector<Section> shoulder={
            {-1.62f,1.18f,.16f,.045f},{-1.39f,1.38f,.31f,.075f},
            {-1.13f,1.44f,.38f,.065f},{-.87f,1.39f,.40f,.08f},
            {-.61f,1.25f,.37f,.12f},{-.32f,1.10f,.285f,.13f},
            {.06f,1.04f,.265f,.13f},{.39f,1.06f,.30f,.145f},
            {.72f,1.17f,.365f,.135f},{1.02f,1.36f,.415f,.10f},
            {1.36f,1.46f,.33f,.055f},{1.65f,1.40f,.17f,.035f}
        };
        Loft(body[Shell],belly);
        Loft(body[Shell],shoulder);
        // A broad near-black inset bridges the lower and upper shell without
        // a chrome tube frame. Only the horizontal reactor remains exposed.
        std::vector<Section> spine={
            {-.65f,.82f,.14f,.10f},{-.47f,.81f,.23f,.17f},
            {-.05f,.76f,.24f,.19f},{.37f,.78f,.22f,.18f},
            {.72f,.99f,.19f,.10f}
        };
        Loft(body[Recess],spine);
        Ellipsoid(body[Rubber],Vec(-.40f,0,1.20f),Vec(.27f,.19f,.045f),32,12);
        for(int side=-1;side<=1;side+=2)
        {
            float s=float(side);
            FairingBand(body[Light],belly,side>0 ? -.10f : Pi+.10f,.15f);
            FairingBand(body[SubtleLight],belly,side>0 ? -.40f : Pi+.40f,.035f);
            FairingBand(body[Light],shoulder,side>0 ? .04f : Pi-.04f,.10f);
            // Long fine engraved seams, kept much darker than the optical bands.
            FairingBand(body[Recess],shoulder,side>0 ? .52f : Pi-.52f,.045f,.002f);
            FairingBand(body[Recess],belly,side>0 ? .72f : Pi-.72f,.035f,.002f);
            // A recessed, longitudinal power unit, framed by a dark side housing.
            Rod(body[Armour],Vec(-.45f,s*.28f,.73f),Vec(.32f,s*.28f,.73f),.15f,.15f,36);
            Rod(body[Metal],Vec(-.35f,s*.28f,.73f),Vec(.23f,s*.28f,.73f),.145f,.145f,36);
            for(int band=0;band<4;++band)
            {
                float x=-.31f+.16f*band;
                std::vector<Vec> ring;
                for(int i=0;i<=40;++i)
                {
                    float a=2*Pi*i/40;
                    ring.push_back(Vec(x,s*.28f+.151f*std::cos(a),.73f+.151f*std::sin(a)));
                }
                Tube(body[Light],ring,.011f,8);
            }
            // The foot rest and hand controls are nestled into the fairing;
            // nothing crosses the large hubless wheel openings.
            Rod(body[Recess],Vec(-.29f,s*.29f,.34f),Vec(-.29f,s*.47f,.34f),.036f,.036f,16);
            Rod(body[Rubber],Vec(1.12f,s*.25f,1.12f),Vec(1.14f,s*.39f,1.12f),.031f,.031f,18);
        }
        for(int index=0;index<2;++index)
        {
            Vec center(index ? FrontAxle : RearAxle,0,WheelRadius);
            float start=(index ? -22.f : 15.f)*Pi/180;
            float finish=(index ? 157.f : 202.f)*Pi/180;
            // Wide integrated shoulders wrap the wheels; no raised motorcycle
            // mudguards, fork rods, axle caps or exposed suspension pistons.
            Lathe(body[Shell],{{-.447f,.668f},{-.451f,.699f},{-.425f,.739f},
                {-.34f,.769f},{-.18f,.783f},{0,.790f},{.18f,.783f},
                {.34f,.769f},{.425f,.739f},{.451f,.699f},{.447f,.668f},
                {.41f,.704f},{.28f,.736f},{0,.747f},{-.28f,.736f},{-.41f,.704f}},
                center,start,finish,112);
            CrownBand(body[Light],center,-.025f,.025f,.795f,start+.02f,finish-.02f);
            for(int side=-1;side<=1;side+=2)
            {
                std::vector<Vec> seam;
                for(int i=0;i<=24;++i)
                {
                    float a=start+(finish-start)*i/24;
                    seam.push_back(center+Vec(.746f*std::cos(a),side*.423f,.746f*std::sin(a)));
                }
                Tube(body[Recess],seam,.005f,6);
            }
        }
        // Vertical nose lights sit flush in the front housing, like the film's
        // narrow slit lamps. White is confined to these functional lights.
        for(int side=-1;side<=1;side+=2)
        {
            Vec center(FrontAxle,0,WheelRadius);
            float y=side*.255f;
            CrownBand(body[Recess],center,y-.038f,y+.038f,.796f,-.28f,.55f);
            CrownBand(body[Headlight],center,y-.013f,y+.013f,.799f,-.23f,.50f);
        }
        // Rear optical spine connects the wheel glow to the emitted light wall.
        CrownBand(body[Light],Vec(RearAxle,0,WheelRadius),-.035f,.035f,.797f,
                  .78f*Pi,1.12f*Pi);
    }

    void BuildRider()
    {
        // An elongated racing tuck fills the long saddle. The shoulder, back
        // and pelvis share one continuous surface instead of separate spheres.
        std::vector<Section> torso={
            {-.60f,1.18f,.11f,.10f},{-.43f,1.25f,.17f,.13f},
            {-.14f,1.35f,.19f,.14f},{.16f,1.43f,.215f,.15f},
            {.43f,1.47f,.22f,.14f},{.61f,1.46f,.135f,.09f}
        };
        Loft(body[Suit],torso);
        Rod(body[Suit],Vec(.58f,0,1.48f),Vec(.73f,0,1.53f),.075f,.065f,20);
        Vec helmet(.84f,0,1.58f),radii(.245f,.151f,.168f);
        Ellipsoid(body[Shell],helmet,radii,48,24);
        Ellipsoid(body[Shell],Vec(.97f,0,1.49f),Vec(.17f,.125f,.08f),32,16);
        Surface(body[Visor],36,14,[&](float u,float v)
        {
            float a=-1.38f+2.76f*u,b=-.50f+.75f*v;
            Vec n(std::cos(b)*std::cos(a),std::cos(b)*std::sin(a),std::sin(b));
            return Vertex(helmet+Vec(n.x*radii.x*1.012f,n.y*radii.y*1.012f,n.z*radii.z*1.012f),
                          Unit(Vec(n.x/radii.x,n.y/radii.y,n.z/radii.z)));
        });
        std::vector<Vec> brow;
        for(int i=0;i<=28;++i)
        {
            float a=-1.33f+2.66f*i/28;
            brow.push_back(helmet+Vec(radii.x*1.023f*std::cos(.25f)*std::cos(a),
                radii.y*1.023f*std::cos(.25f)*std::sin(a),radii.z*std::sin(.25f)));
        }
        Tube(body[SubtleLight],brow,.0055f,8);
        for(int side=-1;side<=1;side+=2)
        {
            float s=float(side);
            FairingBand(body[Light],torso,side>0 ? .30f : Pi-.30f,.12f);
            Vec shoulder(.43f,s*.19f,1.46f),elbow(.75f,s*.29f,1.19f),wrist(1.12f,s*.32f,1.12f);
            Rod(body[Suit],shoulder,elbow,.089f,.072f,22);
            Rod(body[Suit],elbow,wrist,.073f,.055f,22);
            Ellipsoid(body[Suit],shoulder,Vec(.105f,.095f,.095f),24,14);
            Ellipsoid(body[Suit],elbow,Vec(.077f,.077f,.076f),24,14);
            Ellipsoid(body[Rubber],Vec(1.15f,s*.32f,1.11f),Vec(.086f,.062f,.056f),24,14);
            Tube(body[Light],{{.35f,s*.26f,1.49f},{.47f,s*.28f,1.45f},
                {.63f,s*.34f,1.31f},{.76f,s*.36f,1.21f},
                {.95f,s*.37f,1.17f},{1.13f,s*.37f,1.12f}},.008f);
            Vec hip(-.43f,s*.15f,1.19f),knee(.12f,s*.32f,.79f),ankle(-.38f,s*.36f,.39f);
            Rod(body[Suit],hip,knee,.115f,.097f,24);
            Rod(body[Suit],knee,ankle,.086f,.064f,24);
            Ellipsoid(body[Suit],knee,Vec(.11f,.098f,.105f),28,16);
            Ellipsoid(body[Rubber],Vec(-.25f,s*.36f,.33f),Vec(.20f,.09f,.075f),32,16);
            Tube(body[Light],{{-.43f,s*.235f,1.21f},{-.20f,s*.31f,1.04f},
                {.08f,s*.413f,.84f},{.12f,s*.415f,.78f},
                {-.12f,s*.425f,.58f},{-.35f,s*.425f,.41f},
                {-.16f,s*.43f,.345f}},.009f);
            // A broad curved suit panel connects the back stripe to the hips.
            Tube(body[Light],{{-.52f,s*.175f,1.25f},{-.36f,s*.20f,1.36f},
                {-.10f,s*.21f,1.46f},{.15f,s*.22f,1.54f},{.38f,s*.19f,1.58f}},.010f);
        }
        Ellipsoid(body[Recess],Vec(-.07f,0,1.501f),Vec(.115f,.119f,.023f),28,12);
        std::vector<Vec> disc;
        for(int i=0;i<=48;++i)
        {
            float a=2*Pi*i/48;
            disc.push_back(Vec(-.07f+.096f*std::cos(a),.096f*std::sin(a),
                               1.526f+.026f*std::cos(a)));
        }
        Tube(body[Light],disc,.010f,10);
    }
};

inline void SetMaterial(Material material,float r,float g,float b)
{
    GLfloat ambient[4]={0,0,0,1},diffuse[4]={0,0,0,1};
    GLfloat specular[4]={0,0,0,1},emission[4]={0,0,0,1};
    float shine=64;
    if(material==Light || material==SubtleLight || material==Headlight)
    {
        float power=material==SubtleLight ? .35f : 1.f;
        emission[0]=power*r; emission[1]=power*g; emission[2]=power*b;
        if(material==Headlight)
        {
            emission[0]=.77f; emission[1]=.87f; emission[2]=.95f;
        }
    }
    else
    {
        Vec base,highlight;
        switch(material)
        {
        case Shell: base=Vec(.016f,.020f,.025f); highlight=Vec(.74f,.81f,.88f); shine=115; break;
        case Armour: base=Vec(.035f,.042f,.050f); highlight=Vec(.55f,.63f,.70f); shine=90; break;
        case Metal: base=Vec(.13f,.15f,.17f); highlight=Vec(.84f,.91f,.97f); shine=100; break;
        case Rubber: base=Vec(.016f,.019f,.024f); highlight=Vec(.21f,.25f,.30f); shine=31; break;
        case Recess: base=Vec(.005f,.008f,.011f); highlight=Vec(.07f,.10f,.13f); shine=24; break;
        case Suit: base=Vec(.012f,.015f,.019f); highlight=Vec(.17f,.21f,.25f); shine=32; break;
        case Visor: base=Vec(.012f,.031f,.043f); highlight=Vec(.84f,.95f,1.f); shine=120; break;
        default: base=Vec(.04f,.05f,.06f); highlight=Vec(.4f,.4f,.4f); break;
        }
        diffuse[0]=base.x; diffuse[1]=base.y; diffuse[2]=base.z;
        ambient[0]=base.x*.48f; ambient[1]=base.y*.48f; ambient[2]=base.z*.48f;
        // Small reflected neon contributions separate the black shell from the
        // suit and light the recessed metal barrels without bleaching the hue.
        float bounce=material==Metal ? .055f : material==Shell ? .018f : .006f;
        ambient[0]+=r*bounce; ambient[1]+=g*bounce; ambient[2]+=b*bounce;
        specular[0]=highlight.x; specular[1]=highlight.y; specular[2]=highlight.z;
    }
    glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,ambient);
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,diffuse);
    glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,emission);
    glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,shine);
}

inline void Draw(Parts const & parts,float r,float g,float b)
{
    for(int i=0;i<MaterialCount;++i)
        if(!parts[i].vertices.empty())
        {
            SetMaterial(static_cast<Material>(i),r,g,b);
            parts[i].Draw();
        }
}

// CPU geometry stays canonical across screen-mode changes. The optional VBO cache
// is released before context destruction and lazily repopulated by the next Draw.
inline Cache const & Geometry()
{
    static Cache const cache;
    return cache;
}
}

inline void Render(float r,float g,float b,float rearAngle,float frontAngle)
{
    r=std::max(0.f,std::min(1.f,r));
    g=std::max(0.f,std::min(1.f,g));
    b=std::max(0.f,std::min(1.f,b));
    Detail::Cache const & cache=Detail::Geometry();

    GLint oldMatrixMode=GL_MODELVIEW;
    glGetIntegerv(GL_MATRIX_MODE,&oldMatrixMode);
    glPushAttrib(GL_ENABLE_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT | GL_TEXTURE_BIT
                 | GL_COLOR_BUFFER_BIT | GL_POLYGON_BIT | GL_DEPTH_BUFFER_BIT);
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
#ifdef HAVE_GLEW
    GLint oldArrayBuffer=0;
    if(GLEW_VERSION_1_5)
    {
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&oldArrayBuffer);
        glBindBuffer(GL_ARRAY_BUFFER,0);
    }
    if(GLEW_VERSION_1_3)
    {
        GLint units=1;
        glGetIntegerv(GL_MAX_TEXTURE_UNITS,&units);
        for(GLint unit=0;unit<units;++unit)
        {
            glActiveTexture(GL_TEXTURE0+unit);
            glDisable(GL_TEXTURE_1D);
            glDisable(GL_TEXTURE_2D);
            glDisable(GL_TEXTURE_3D);
            glDisable(GL_TEXTURE_CUBE_MAP);
            glDisable(GL_TEXTURE_GEN_S);
            glDisable(GL_TEXTURE_GEN_T);
            glDisable(GL_TEXTURE_GEN_R);
            glDisable(GL_TEXTURE_GEN_Q);
            glClientActiveTexture(GL_TEXTURE0+unit);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        }
        glActiveTexture(GL_TEXTURE0);
        glClientActiveTexture(GL_TEXTURE0);
    }
    if(GLEW_VERSION_1_4)
    {
        glDisableClientState(GL_SECONDARY_COLOR_ARRAY);
        glDisableClientState(GL_FOG_COORD_ARRAY);
    }
#endif
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);
    glDisableClientState(GL_EDGE_FLAG_ARRAY);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    Detail::Draw(cache.body,r,g,b);
    for(int wheel=0;wheel<2;++wheel)
    {
        glPushMatrix();
        glTranslatef(wheel ? FrontAxle : RearAxle,0,WheelRadius);
        glRotatef(-(wheel ? frontAngle : rearAngle)*180.f/Detail::Pi,0,1,0);
        Detail::Draw(cache.wheel,r,g,b);
        glPopMatrix();
    }

    glPopMatrix();
    glMatrixMode(oldMatrixMode);
#ifdef HAVE_GLEW
    if(GLEW_VERSION_1_5)
        glBindBuffer(GL_ARRAY_BUFFER,oldArrayBuffer);
#endif
    glPopClientAttrib();
    glPopAttrib();
}
#endif // DEDICATED
}
#endif
