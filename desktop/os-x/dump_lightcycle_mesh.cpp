// CPU-only export of the renderer's canonical mesh. This is deliberately not a
// game target: export_lightcycle_blender.py compiles it with the native config.
#include "gLightcycleModel.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace model = gLightcycleModel;
namespace detail = gLightcycleModel::Detail;

struct Summary
{
    std::size_t vertices = 0;
    detail::Vec low = detail::Vec(INFINITY, INFINITY, INFINITY);
    detail::Vec high = detail::Vec(-INFINITY, -INFINITY, -INFINITY);
    float normalError = 0;

    void Add(detail::Parts const & parts, detail::Vec const & offset = detail::Vec())
    {
        for (detail::Mesh const & mesh : parts)
        {
            if (mesh.vertices.size() % 3)
                throw std::runtime_error("Native mesh has an incomplete triangle");
            for (detail::Vertex const & vertex : mesh.vertices)
            {
                detail::Vec p = vertex.position + offset;
                detail::Vec n = vertex.normal;
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)
                    || !std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z))
                    throw std::runtime_error("Native mesh contains a nonfinite vertex/normal");
                normalError = std::max(normalError, std::fabs(std::sqrt(detail::Dot(n,n)) - 1.f));
                low.x = std::min(low.x,p.x); low.y = std::min(low.y,p.y); low.z = std::min(low.z,p.z);
                high.x = std::max(high.x,p.x); high.y = std::max(high.y,p.y); high.z = std::max(high.z,p.z);
                ++vertices;
            }
        }
        if (normalError > .001f)
            throw std::runtime_error("Native mesh normals are not unit length");
    }

    void Write() const
    {
        std::cout << "{\"vertices\":" << vertices << ",\"triangles\":" << vertices/3
                  << ",\"bounds\":[[" << low.x << ',' << low.y << ',' << low.z << "],["
                  << high.x << ',' << high.y << ',' << high.z << "]]"
                  << ",\"max_normal_error\":" << normalError << '}';
    }
};

void WriteParts(detail::Parts const & parts)
{
    std::cout << '[';
    for (int material=0;material<detail::MaterialCount;++material)
    {
        if (material)
            std::cout << ',';
        std::cout << "{\"material\":" << material << ",\"vertices\":[";
        bool first=true;
        for (detail::Vertex const & vertex : parts[material].vertices)
        {
            if (!first)
                std::cout << ',';
            first=false;
            std::cout << '[' << vertex.position.x << ',' << vertex.position.y << ',' << vertex.position.z
                      << ',' << vertex.normal.x << ',' << vertex.normal.y << ',' << vertex.normal.z << ']';
        }
        std::cout << "]}";
    }
    std::cout << ']';
}

int main()
{
    try
    {
        static_assert(detail::MaterialCount==10,"Update exporter material names for the native material enum");
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        detail::Cache const & cache=detail::Geometry();
        Summary body,rider,wheel,complete;
        body.Add(cache.body);
        rider.Add(cache.rider);
        wheel.Add(cache.wheel);
        complete.Add(cache.body);
        complete.Add(cache.rider);
        complete.Add(cache.wheel,detail::Vec(model::RearAxle,0,model::WheelRadius));
        complete.Add(cache.wheel,detail::Vec(model::FrontAxle,0,model::WheelRadius));
        std::cout << "{\"schema_version\":1,\"axes\":{\"forward\":\"+X\",\"axle\":\"Y\",\"up\":\"+Z\"},"
                     "\"constants\":{\"rear_axle\":" << model::RearAxle << ",\"front_axle\":" << model::FrontAxle
                  << ",\"wheel_radius\":" << model::WheelRadius << "},"
                     "\"materials\":[\"Shell\",\"Armour\",\"Metal\",\"Rubber\",\"Recess\",\"Suit\",\"Visor\",\"Light\",\"SubtleLight\",\"Headlight\"],"
                     "\"summary\":{\"body\":";
        body.Write(); std::cout << ",\"rider\":"; rider.Write();
        std::cout << ",\"wheel\":"; wheel.Write();
        std::cout << ",\"complete\":"; complete.Write();
        std::cout << "},\"parts\":{\"body\":";
        WriteParts(cache.body);
        std::cout << ",\"rider\":";
        WriteParts(cache.rider);
        std::cout << ",\"wheel\":";
        WriteParts(cache.wheel);
        std::cout << "}}\n";
        return 0;
    }
    catch (std::exception const & error)
    {
        std::cerr << "Native lightcycle export: " << error.what() << '\n';
        return 1;
    }
}
