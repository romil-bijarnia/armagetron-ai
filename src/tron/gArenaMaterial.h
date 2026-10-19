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
        int const size = 512;
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
                if (kind_ == Floor)
                {
                    float dx = std::min(x % 128, 128 - x % 128);
                    float dy = std::min(y % 128, 128 - y % 128);
                    float minor = std::max(Line(dx, 1.5f), Line(dy, 1.5f));
                    float major = std::max(Line(std::min(x, size - x), 2.f),
                                           Line(std::min(y, size - y), 2.f));
                    float junction = Line(dx, 3.f) * Line(dy, 3.f);
                    intensity = .105f + .19f * minor + .12f * major + .17f * junction;
                }
                else if (kind_ == Trail)
                {
                    // A bright top edge marks the lethal trail. The body stays
                    // opaque and smoothly shaded instead of repeating arrows.
                    float v = float(y) / (size - 1);
                    float top = std::exp(-v * 34.f);
                    float bottom = std::exp(-(1.f - v) * 42.f);
                    intensity = .34f + .16f * (1.f - v) + .50f * top + .28f * bottom;
                }
                else
                {
                    // Near-black perimeter panels with faint seams and one rail
                    // along the floor bright enough for the glow pass to pick up.
                    float v = float(y) / (size - 1);
                    float seam = Line(std::min(x % 128, 128 - x % 128), 2.f);
                    float rail = std::fabs(v - .97f);
                    intensity = .07f + .03f * v + .05f * seam
                                + .95f * Line(rail, .004f) + .22f * Line(rail, .02f);
                }
                Uint8 light = Uint8(std::min(1.f, intensity) * 255.f + .5f);
                row[x] = SDL_MapRGBA(raw->format, light, light, light, 255);
            }
        }
        Upload(surface);
    }
};
#endif
#endif
