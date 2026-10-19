#ifndef ARMAGETRON_ARENA_MATERIAL_H
#define ARMAGETRON_ARENA_MATERIAL_H

#include "gStuff.h"

#ifndef DEDICATED
#include "rTexture.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Procedural materials use the normal texture cache and context-loss handling.
class gArenaMaterial : public rISurfaceTexture
{
public:
    enum Kind { Floor, Trail, Rim };
    explicit gArenaMaterial(Kind kind)
        : rISurfaceTexture(kind == Floor ? rTextureGroups::TEX_FLOOR : rTextureGroups::TEX_WALL,
                          true, kind == Floor, true), kind_(kind) {}

private:
    Kind kind_;
    class Surface : public rSurface
    {
    public:
        explicit Surface(SDL_Surface * surface) { Create(surface); }
    };

    static float Line(float distance, float width)
    {
        return std::max(0.f, 1.f - distance / width);
    }

    void OnSelectCore() override
    {
        // Preserve narrow panel edges and rounded corners at Retina resolution.
        int const size = kind_ == Floor ? 2048 : 1024;
        SDL_Surface * raw = SDL_CreateRGBSurface(SDL_SWSURFACE, size, size, 32,
                                                0x000000ff, 0x0000ff00,
                                                0x00ff0000, 0xff000000);
        if (!raw)
            throw std::runtime_error("Could not allocate arena material");
        Surface surface(raw);
        for (int y = 0; y < size; ++y)
        {
            Uint32 * row = reinterpret_cast<Uint32 *>(static_cast<Uint8 *>(raw->pixels) + y * raw->pitch);
            for (int x = 0; x < size; ++x)
            {
                float intensity;
                Uint8 alpha = 255;
                if (kind_ == Floor)
                {
                    // Broad rounded circuit panels leave quiet black glass for
                    // the reflected bikes and trails instead of a dense wire grid.
                    float px = std::fabs(float(x) * 512.f / size - 256.f);
                    float py = std::fabs(float(y) * 512.f / size - 256.f);
                    float qx = px - 205.f, qy = py - 205.f;
                    float outside = std::sqrt(std::max(0.f,qx)*std::max(0.f,qx)
                                              + std::max(0.f,qy)*std::max(0.f,qy));
                    float edge = outside + std::min(std::max(qx,qy),0.f) - 27.f;
                    float seam = Line(std::fabs(edge), 1.7f);
                    float shoulder = Line(std::fabs(edge), 5.f);
                    float inner = Line(std::fabs(edge+10.f), .9f);
                    float marker = std::max(Line(px,.8f)*Line(py,4.f),
                                            Line(py,.8f)*Line(px,4.f));
                    intensity = .043f + .50f*seam + .07f*shoulder + .035f*inner + .065f*marker;
                }
                else if (kind_ == Trail)
                {
                    // A transparent energy sheet between two narrow optical
                    // edges. Greyscale modulation preserves every player's hue;
                    // the field adds light without painting an opaque panel.
                    float v = float(y) / (size - 1);
                    float u = float(x) / size;
                    float top = std::exp(-v * 85.f);
                    float bottom = std::exp(-(1.f - v) * 60.f);
                    float shoulder = std::exp(-v * 14.f);
                    float foot = std::exp(-(1.f - v) * 10.f);
                    float weave = .5f + .5f * std::sin(v * 452.38934f
                                      + .55f * std::sin(u * 6.2831853f));
                    float edge = std::max(top,bottom);
                    intensity = .60f + .025f*weave + .12f*shoulder + .09f*foot + .35f*edge;
                    float opacity = .27f + .014f*weave + .13f*shoulder + .10f*foot + .72f*edge;
                    alpha = Uint8(std::min(1.f,opacity)*255.f+.5f);
                }
                else
                {
                    // Architectural panels with a bright lower rail and a
                    // restrained second seam above it.
                    float v = float(y) / (size - 1);
                    float panel = std::fmod(float(x) * 512.f / size, 128.f);
                    float seam = Line(std::min(panel, 128.f - panel), 2.f);
                    float rail = std::fabs(v - .985f);
                    float upper = std::fabs(v - .93f);
                    intensity = .035f + .02f * v + .025f * seam
                                + .94f * Line(rail, .0035f) + .16f * Line(rail, .016f)
                                + .19f * Line(upper, .0025f);
                }
                Uint8 light = Uint8(std::min(1.f, intensity) * 255.f + .5f);
                row[x] = SDL_MapRGBA(raw->format, light, light, light, alpha);
            }
        }
        Upload(surface);
    }
};
#endif
#endif
