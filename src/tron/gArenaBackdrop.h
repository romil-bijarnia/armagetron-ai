#ifndef ARMAGETRON_ARENA_BACKDROP_H
#define ARMAGETRON_ARENA_BACKDROP_H

#include "rRender.h"

#ifndef DEDICATED
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

// A visual stadium, wholly outside the physical rim. Geometry is retained in CPU
// memory so it survives context recreation and can be used by the mirrored pass.
// Coordinates remain in world space: the caller owns all model/view transforms.
class gArenaBackdrop
{
public:
    static void RenderSegment(float x1, float y1, float x2, float y2,
                              float outwardX, float outwardY,
                              float r, float g, float b,
                              bool miterStart=false, bool miterEnd=false)
    {
        Key key = {{x1, y1, x2, y2, outwardX, outwardY, r, g, b,
                    float(miterStart),float(miterEnd)}};
        for (float value : key)
            if (!std::isfinite(value))
                return;
        float dx = x2 - x1, dy = y2 - y1;
        float length = std::hypot(dx, dy);
        if (!std::isfinite(length) || length < .001f)
            return;
        // Stable stationing makes ribs and lamps meet across collinear rim splits.
        if (dx < 0 || (dx == 0 && dy < 0))
        {
            std::swap(x1, x2);
            std::swap(y1, y2);
            std::swap(miterStart,miterEnd);
            dx = -dx;
            dy = -dy;
        }
        float tx = dx / length, ty = dy / length;
        // Accept an approximate normal, but never let the architecture drift inward.
        float side = tx * outwardY - ty * outwardX;
        if (std::fabs(side) < .00001f)
            return;
        float ox = side > 0 ? -ty : ty, oy = side > 0 ? tx : -tx;
        r = Clamp(r); g = Clamp(g); b = Clamp(b);
        key = {{x1, y1, x2, y2, ox, oy, r, g, b,float(miterStart),float(miterEnd)}};
        Cache & cache = Storage();
        auto found = cache.meshes.find(key);
        if (found == cache.meshes.end())
        {
            Mesh mesh;
            Build(mesh, x1, y1, tx, ty, ox, oy, length, r, g, b,miterStart,miterEnd);
            cache.vertices += mesh.Count();
            found = cache.meshes.emplace(key, std::move(mesh)).first;
            // Rim tessellation and custom-map changes cannot grow the cache forever.
            // Eviction happens only when a new segment is first encountered.
            while ((cache.meshes.size() > 512 || cache.vertices > 1500000) && cache.meshes.size() > 1)
            {
                auto oldest = cache.meshes.end();
                for (auto it = cache.meshes.begin(); it != cache.meshes.end(); ++it)
                    if (it != found && (oldest == cache.meshes.end() || it->second.used < oldest->second.used))
                        oldest = it;
                cache.vertices -= oldest->second.Count();
                cache.meshes.erase(oldest);
            }
        }
        Mesh & mesh = found->second;
        mesh.used = ++cache.clock;
        RenderEnd();
        State state;
        Draw(mesh.structure);
        Draw(mesh.lights);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);
        Draw(mesh.halos);
    }

private:
    struct Point { float s, d, z; };
    struct Colour { float r, g, b, a; };
    struct Vertex { float x, y, z, r, g, b, a; };
    using Batch = std::vector<Vertex>;
    using Key = std::array<float, 11>;
    struct Mesh
    {
        Batch structure, lights, halos;
        std::uint64_t used = 0;
        std::size_t Count() const { return structure.size() + lights.size() + halos.size(); }
    };
    struct Cache
    {
        std::map<Key, Mesh> meshes;
        std::size_t vertices = 0;
        std::uint64_t clock = 0;
    };
    struct Frame
    {
        float x, y, tx, ty, ox, oy, length;
        bool miterStart, miterEnd;
        Vertex Make(Point p, Colour c) const
        {
            float along=p.s;
            // At a true rectangular corner the adjoining profiles meet on the
            // outward 45-degree bisector. Other segment ends remain untouched.
            // Only boundary vertices move; world-stationed details use a plain
            // frame so there is no stretch or duplicate lamp population.
            if (miterStart && p.s==0) along-=p.d;
            if (miterEnd && p.s==length) along+=p.d;
            return {x + tx * along + ox * p.d, y + ty * along + oy * p.d,
                    p.z, c.r, c.g, c.b, c.a};
        }
    };

    static Cache & Storage() { static Cache cache; return cache; }
    static float Clamp(float v) { return std::max(0.f, std::min(1.f, v)); }
    static Colour Scale(Colour c, float gain)
    {
        return {c.r * gain, c.g * gain, c.b * gain, c.a};
    }
    static void Quad(Batch & batch, Frame const & f, Point a, Point b, Point c, Point d,
                     Colour lower, Colour upper)
    {
        Vertex va = f.Make(a, lower), vb = f.Make(b, lower);
        Vertex vc = f.Make(c, upper), vd = f.Make(d, upper);
        batch.insert(batch.end(), {va, vb, vc, va, vc, vd});
    }
    static void Box(Batch & batch, Frame const & f, float s0, float s1,
                    float d0, float d1, float z0, float z1, Colour colour)
    {
        s0 = std::max(0.f, s0); s1 = std::min(f.length, s1);
        if (s0 >= s1)
            return;
        Colour shade = Scale(colour, .64f), top = Scale(colour, 1.28f);
        Quad(batch, f, {s0,d0,z0}, {s1,d0,z0}, {s1,d0,z1}, {s0,d0,z1}, shade, colour);
        Quad(batch, f, {s1,d1,z0}, {s0,d1,z0}, {s0,d1,z1}, {s1,d1,z1}, shade, shade);
        if (!(f.miterStart && s0==0))
            Quad(batch, f, {s0,d1,z0}, {s0,d0,z0}, {s0,d0,z1}, {s0,d1,z1}, shade, colour);
        if (!(f.miterEnd && s1==f.length))
            Quad(batch, f, {s1,d0,z0}, {s1,d1,z0}, {s1,d1,z1}, {s1,d0,z1}, shade, colour);
        Quad(batch, f, {s0,d0,z1}, {s1,d0,z1}, {s1,d1,z1}, {s0,d1,z1}, top, colour);
        Quad(batch, f, {s0,d1,z0}, {s1,d1,z0}, {s1,d0,z0}, {s0,d0,z0}, shade, shade);
    }
    // An extruded cross-section: its outward-monotone upper profile gives closed
    // end caps without a polygon tessellator or assumptions about arena shape.
    static void Shell(Batch & batch, Frame const & f, std::vector<Point> const & profile, Colour colour)
    {
        for (std::size_t i = 1; i < profile.size(); ++i)
        {
            Point a = profile[i-1], b = profile[i];
            bool rising = b.z > a.z + .01f;
            Colour face = Scale(colour, rising ? .77f : 1.24f);
            Quad(batch, f, {0,a.d,a.z}, {f.length,a.d,a.z},
                 {f.length,b.d,b.z}, {0,b.d,b.z}, Scale(face,.83f), face);
            if (b.d > a.d && (a.z > 0 || b.z > 0))
            {
                if (!f.miterStart)
                    Quad(batch, f, {0,a.d,0}, {0,b.d,0}, {0,b.d,b.z}, {0,a.d,a.z},
                         Scale(colour,.64f), Scale(colour,.91f));
                if (!f.miterEnd)
                    Quad(batch, f, {f.length,b.d,0}, {f.length,a.d,0},
                         {f.length,a.d,a.z}, {f.length,b.d,b.z},
                         Scale(colour,.64f), Scale(colour,.91f));
            }
        }
    }
    static void Beam(Batch & batch, Frame const & f, float s, float halfWidth,
                     float d0, float z0, float d1, float z1, float thickness, Colour colour)
    {
        float dd = d1 - d0, dz = z1 - z0, inverse = thickness / std::hypot(dd, dz);
        float nd = -dz * inverse, nz = dd * inverse;
        float s0 = std::max(0.f, s-halfWidth), s1 = std::min(f.length, s+halfWidth);
        if (s0 >= s1)
            return;
        Point p[] = {{0,d0-nd,z0-nz}, {0,d1-nd,z1-nz},
                     {0,d1+nd,z1+nz}, {0,d0+nd,z0+nz}};
        for (int i = 0; i < 4; ++i)
        {
            Point a = p[i], b = p[(i+1)%4];
            Quad(batch, f, {s0,a.d,a.z}, {s1,a.d,a.z}, {s1,b.d,b.z}, {s0,b.d,b.z},
                 Scale(colour,.72f), colour);
        }
        Quad(batch, f, {s0,p[0].d,p[0].z}, {s0,p[1].d,p[1].z},
             {s0,p[2].d,p[2].z}, {s0,p[3].d,p[3].z}, colour, Scale(colour,.7f));
        Quad(batch, f, {s1,p[3].d,p[3].z}, {s1,p[2].d,p[2].z},
             {s1,p[1].d,p[1].z}, {s1,p[0].d,p[0].z}, colour, Scale(colour,.7f));
    }
    static void Light(Mesh & mesh, Frame const & f, float s0, float s1,
                      float depth, float height, float thickness, Colour colour, float halo)
    {
        s0 = std::max(0.f,s0); s1 = std::min(f.length,s1);
        if (s0 >= s1)
            return;
        Quad(mesh.lights, f, {s0,depth,height-thickness}, {s1,depth,height-thickness},
             {s1,depth,height+thickness}, {s0,depth,height+thickness}, colour, colour);
        if (halo <= 0)
            return;
        // Narrow mesh halos have bounded screen coverage; bloom supplies the soft
        // optical spill. No camera-facing quads or per-frame billboard work.
        Colour soft = colour;
        soft.a = halo;
        Quad(mesh.halos, f, {s0,depth-.015f,height-thickness*5},
             {s1,depth-.015f,height-thickness*5}, {s1,depth-.015f,height+thickness*5},
             {s0,depth-.015f,height+thickness*5}, soft, soft);
    }
    static void Terrace(std::vector<Point> & profile, float startDepth, float startHeight,
                        float tread, float rise, int rows)
    {
        profile.push_back({0,startDepth,startHeight});
        for (int row = 0; row < rows; ++row)
        {
            float d = startDepth + row*tread, z = startHeight + row*rise;
            // Small chamfers keep the silhouette architectural rather than serrated.
            profile.push_back({0,d+.12f,z+rise*.9f});
            profile.push_back({0,d+tread-.10f,z+rise*.9f});
            profile.push_back({0,d+tread,z+rise});
        }
    }
    static void Build(Mesh & mesh, float x, float y, float tx, float ty, float ox, float oy,
                      float length, float r, float g, float b,bool miterStart,bool miterEnd)
    {
        Frame f = {x,y,tx,ty,ox,oy,length,miterStart,miterEnd};
        Frame details = {x,y,tx,ty,ox,oy,length,false,false};
        Colour graphite = {.022f+.008f*r, .034f+.010f*g, .047f+.012f*b, 1};
        Colour steel = {.045f+.012f*r, .069f+.016f*g, .085f+.020f*b, 1};
        Colour cyan = {.23f+.07f*r, .78f+.12f*g, .94f+.06f*b, 1};
        Colour dim = {.08f, .25f+.07f*g, .33f+.07f*b, 1};
        Colour white = {.70f,.92f,1.f,1};
        std::size_t bays=static_cast<std::size_t>(std::min(130.f,std::ceil(length/14.f)+2));
        mesh.structure.reserve(1248+180*bays);
        mesh.lights.reserve(42+48*bays);
        mesh.halos.reserve(24+6*bays);

        std::vector<Point> bowl = {{0,3.15f,0}, {0,3.15f,3.45f}, {0,5.f,3.45f}};
        Terrace(bowl,5.f,3.45f,1.35f,.94f,6);
        bowl.push_back({0,15.9f,9.09f});
        Terrace(bowl,16.f,9.65f,1.35f,1.14f,6);
        bowl.push_back({0,26.9f,16.49f});
        Terrace(bowl,27.f,17.1f,1.4f,1.14f,5);
        bowl.push_back({0,34.2f,25.25f});
        bowl.push_back({0,35.5f,25.55f});
        bowl.push_back({0,36.2f,24.9f});
        bowl.push_back({0,37.f,0});
        bowl.push_back(bowl.front());
        Shell(mesh.structure,f,bowl,graphite);

        // Recessed walkways, with only the edge rail exposed to the playing field.
        float railDepth[] = {3.09f,13.30f,24.30f};
        float railHeight[] = {3.95f,9.65f,17.05f};
        for (int tier=0;tier<3;++tier)
        {
            float d=railDepth[tier], z=railHeight[tier];
            Box(mesh.structure,f,0,length,d+.04f,d+.40f,z-.23f,z+.12f,steel);
            Light(mesh,f,0,length,d,z,.035f,Scale(cyan,tier==0 ? .82f : .62f),.10f);
            Light(mesh,f,0,length,d+.02f,z-.19f,.012f,dim,0);
        }

        // A thin swept canopy breaks the skyline without enclosing the arena.
        std::vector<Point> canopy = {{0,28.65f,26.9f}, {0,28.8f,27.3f},
            {0,36.8f,27.85f}, {0,38.4f,27.55f}, {0,38.4f,27.20f},
            {0,35.f,27.4f}, {0,29.f,26.9f}, {0,28.65f,26.9f}};
        // Only the canopy skin is extruded: its underside must stay open above
        // the tiers rather than acquire the bowl's ground-reaching end caps.
        for (std::size_t i=1;i<canopy.size();++i)
        {
            Point a=canopy[i-1], c=canopy[i];
            Quad(mesh.structure,f,{0,a.d,a.z},{length,a.d,a.z},{length,c.d,c.z},{0,c.d,c.z},
                 Scale(steel,.62f),Scale(steel,.9f));
        }
        Light(mesh,f,0,length,28.62f,27.04f,.035f,Scale(cyan,.75f),.12f);

        // Station spacing is fixed on ordinary maps and grows only for extreme
        // custom segments, bounding mesh size even with kilometre-long rims.
        float stride=14.f*std::max(1.f,std::ceil(length/(14.f*128.f)));
        float origin=x*tx+y*ty;
        double first=std::ceil((double(origin)-1.)/stride)*stride-origin;
        for (double station=first;station<double(length)+1.;station+=stride)
        {
            float s=float(station);
            Box(mesh.structure,details,s-.32f,s+.32f,3.035f,4.2f,.05f,4.1f,steel);
            Box(mesh.structure,details,s-.42f,s+.42f,14.2f,15.4f,.3f,10.7f,steel);
            Beam(mesh.structure,details,s,.31f,14.8f,8.8f,30.3f,26.95f,.23f,steel);
            Box(mesh.structure,details,s-.40f,s+.40f,33.8f,35.f,22.5f,27.4f,steel);
            // Warm-white lamps are orderly architectural fixtures, not a field
            // of randomly coloured neon or animated spectator lights.
            Box(mesh.structure,details,s-2.45f,s+2.45f,28.53f,29.f,26.38f,27.40f,graphite);
            Light(mesh,details,s-2.15f,s+2.15f,28.49f,26.89f,.32f,white,.08f);
            Light(mesh,details,s-.32f,s+.32f,3.015f,1.55f,.055f,Scale(cyan,.72f),0);
        }

        // Sparse aisle markers give the dark seating scale. They follow a quiet,
        // repeated arrangement and sit above the actual terrace surfaces.
        float tierDepth[] = {5.f,16.f,27.f};
        float tierHeight[] = {3.45f,9.65f,17.1f};
        float rise[] = {.94f,1.14f,1.14f};
        for (int tier=0;tier<3;++tier)
            for (int row=1;row<5;row+=2)
            {
                float tread=tier==2 ? 1.4f : 1.35f;
                float d=tierDepth[tier]+row*tread+.10f;
                float z=tierHeight[tier]+(row+.9f)*rise[tier]+.035f;
                float offset=(tier+row)%2 ? stride*.5f : 0;
                double start=std::ceil((double(origin)-offset-.075)/stride)*stride+offset-origin;
                for (double station=start;station<double(length)+.075;station+=stride)
                    Light(mesh,details,float(station)-.075f,float(station)+.075f,d,z,.035f,
                          Scale(dim,.78f),0);
            }
    }

    // The fixed-function path is intentional: untextured architectural colours
    // are already baked into the mesh, and any caller's material shader is restored.
    class State
    {
    public:
        State()
        {
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT |
                         GL_DEPTH_BUFFER_BIT | GL_POLYGON_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT);
            glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
#ifdef HAVE_GLEW
            shaders_=GLEW_VERSION_2_0;
            buffers_=GLEW_VERSION_1_5;
            multitexture_=GLEW_VERSION_1_3;
            if (shaders_)
            {
                glGetIntegerv(GL_CURRENT_PROGRAM,&program_);
                glUseProgram(0);
            }
            if (buffers_)
            {
                glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&arrayBuffer_);
                glBindBuffer(GL_ARRAY_BUFFER,0);
            }
            if (multitexture_)
            {
                glGetIntegerv(GL_ACTIVE_TEXTURE,&texture_);
                glGetIntegerv(GL_CLIENT_ACTIVE_TEXTURE,&clientTexture_);
                GLint units=1;
                glGetIntegerv(GL_MAX_TEXTURE_UNITS,&units);
                for (GLint i=0;i<units;++i)
                {
                    glActiveTexture(GL_TEXTURE0+i);
                    glClientActiveTexture(GL_TEXTURE0+i);
                    DisableTexture();
                    glDisable(GL_TEXTURE_3D);
                    glDisable(GL_TEXTURE_CUBE_MAP);
                }
                glActiveTexture(GL_TEXTURE0);
                glClientActiveTexture(GL_TEXTURE0);
            }
            else
#endif
                DisableTexture();
            glDisable(GL_LIGHTING);
            glDisable(GL_ALPHA_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_CULL_FACE);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_TRUE);
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
            glShadeModel(GL_SMOOTH);
            glDisableClientState(GL_NORMAL_ARRAY);
            glDisableClientState(GL_INDEX_ARRAY);
            glDisableClientState(GL_EDGE_FLAG_ARRAY);
#ifdef HAVE_GLEW
            if (GLEW_VERSION_1_4)
            {
                glDisableClientState(GL_SECONDARY_COLOR_ARRAY);
                glDisableClientState(GL_FOG_COORD_ARRAY);
            }
#endif
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);
        }
        ~State()
        {
            glPopClientAttrib();
            glPopAttrib();
#ifdef HAVE_GLEW
            if (buffers_) glBindBuffer(GL_ARRAY_BUFFER,arrayBuffer_);
            if (shaders_) glUseProgram(program_);
            if (multitexture_)
            {
                glActiveTexture(texture_);
                glClientActiveTexture(clientTexture_);
            }
#endif
        }
    private:
        static void DisableTexture()
        {
            glDisable(GL_TEXTURE_1D);
            glDisable(GL_TEXTURE_2D);
            glDisable(GL_TEXTURE_GEN_S);
            glDisable(GL_TEXTURE_GEN_T);
            glDisable(GL_TEXTURE_GEN_R);
            glDisable(GL_TEXTURE_GEN_Q);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        }
#ifdef HAVE_GLEW
        GLint program_=0, arrayBuffer_=0, texture_=GL_TEXTURE0, clientTexture_=GL_TEXTURE0;
        bool shaders_=false, buffers_=false, multitexture_=false;
#endif
    };
    static void Draw(Batch const & vertices)
    {
        if (vertices.empty())
            return;
        glVertexPointer(3,GL_FLOAT,sizeof(Vertex),&vertices.front().x);
        glColorPointer(4,GL_FLOAT,sizeof(Vertex),&vertices.front().r);
        glDrawArrays(GL_TRIANGLES,0,static_cast<GLsizei>(vertices.size()));
    }
};
#endif
#endif
