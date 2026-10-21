#ifndef ARMAGETRON_LIGHT_COLOUR_H
#define ARMAGETRON_LIGHT_COLOUR_H

// Shared display-RGB emission policy for bike inlays and lightwalls. A cyan
// light may have a white optical core; a pure blue light must stay blue. Keep
// saturated warm emission in gamut before the legacy 8-bit scene target clips
// individual channels. Bloom applies its own colour-preserving shoulder later.
namespace gLightColour
{
inline char const * ShaderSource()
{
    return R"GLSL(#version 120
float cyanLight(vec3 tint) {
    float peak=max(max(tint.r,tint.g),max(tint.b,0.00001));
    return smoothstep(0.35,0.70,(min(tint.g,tint.b)-tint.r)/peak);
}
vec3 lightCore(vec3 tint,float heat) {
    float peak=max(max(tint.r,tint.g),max(tint.b,0.00001));
    float cyan=cyanLight(tint);
    float warm=smoothstep(0.35,0.65,(tint.r-tint.b)/peak);
    vec3 warmCore=vec3(peak,min(peak,tint.g+peak*0.12),tint.b);
    vec3 target=mix(vec3(peak),warmCore,warm);
    return mix(tint,target,mix(0.08,0.68,cyan)*clamp(heat,0.0,1.0));
}
vec3 boundedLight(vec3 emission,vec3 tint) {
    float peak=max(emission.r,max(emission.g,emission.b));
    float scale=min(1.0,1.0/max(peak,0.00001));
    return emission*mix(scale,1.0,cyanLight(tint));
}
)GLSL";
}
}

#endif
