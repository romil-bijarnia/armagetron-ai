#ifndef ARMAGETRON_LIGHTCYCLE_MATERIAL_H
#define ARMAGETRON_LIGHTCYCLE_MATERIAL_H

#include "aa_config.h"

#ifndef DEDICATED
#include "rGL.h"
#include "rScreen.h"
#include "tConsole.h"
#include <algorithm>
#include <cmath>

// Smooth materials for the sculpted lightcycle. Geometry continues to provide
// ordinary GL materials and normals, so machines without shaders use lighting
// from the compatibility renderer. Emissive trim keeps its exact player colour.
class gLightcycleMaterial
{
#ifdef HAVE_GLEW
    static GLuint & Program() { static GLuint value = 0; return value; }
    static bool & Failed() { static bool value = false; return value; }
    struct LightBindings
    {
        GLint colour=-1, axis=-1, shape=-1, rear=-1, front=-1, reactorA=-1, reactorB=-1;
    };
    static LightBindings & Lights() { static LightBindings value; return value; }

    static void Unload()
    {
        if (Program()) glDeleteProgram(Program());
        Program() = 0;
        Failed() = false;
        Lights() = LightBindings();
    }

    static GLuint Compile(GLenum type, char const * source)
    {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);
        GLint success = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            char message[1024] = "";
            glGetShaderInfoLog(shader, sizeof(message) - 1, NULL, message);
            con << "Lightcycle material: " << message << "\n";
            glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    static bool Prepare()
    {
        static rCallbackBeforeScreenModeChange cleanup(&Unload);
        if (Program()) return true;
        if (Failed() || !GLEW_VERSION_2_0) return false;

        char const * vertex =
            "#version 120\n"
            "varying vec3 viewPosition;\n"
            "varying vec3 viewNormal;\n"
            "void main() {\n"
            "    vec4 p = gl_ModelViewMatrix * gl_Vertex;\n"
            "    viewPosition = p.xyz;\n"
            "    viewNormal = gl_NormalMatrix * gl_Normal;\n"
            "    gl_ClipVertex = p;\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        char const * fragment =
            "#version 120\n"
            "varying vec3 viewPosition;\n"
            "varying vec3 viewNormal;\n"
            "uniform vec3 neonColour;\n"
            "uniform vec3 cycleAxis;\n"
            "uniform vec3 ringShape;\n" // radius, half width, local light reach
            "uniform vec3 rearLight;\n"
            "uniform vec3 frontLight;\n"
            "uniform vec3 reactorA;\n"
            "uniform vec3 reactorB;\n"
            "float highlight(vec3 n, vec3 v, vec3 l, float power) {\n"
            "    return pow(max(dot(n, normalize(v + l)), 0.0), power);\n"
            "}\n"
            // Evaluate the nearest emitting part of an annulus/strip, not a
            // point at the empty wheel hub. The light follows the real bike in
            // normal views and in the mirrored floor pass.
            "vec3 neonSample(vec3 source, vec3 n, vec3 v, float reach) {\n"
            "    vec3 delta = source - viewPosition;\n"
            "    float distance2 = dot(delta, delta);\n"
            "    float distance = sqrt(max(distance2, 0.000001));\n"
            "    vec3 l = delta / distance;\n"
            "    float attenuation = 1.0 / (1.0 + distance2 / (reach * reach));\n"
            "    attenuation *= 1.0 - smoothstep(reach * 1.5, reach * 3.0, distance);\n"
            "    float diffuse = max(dot(n, l), 0.0);\n"
            "    float shine = highlight(n, v, l, max(8.0, gl_FrontMaterial.shininess * 0.45));\n"
            "    vec3 response = gl_FrontMaterial.diffuse.rgb * (1.7 * diffuse);\n"
            "    response += gl_FrontMaterial.specular.rgb * (0.075 + 0.62 * shine);\n"
            "    return neonColour * response * attenuation;\n"
            "}\n"
            "vec3 ringLight(vec3 center, vec3 n, vec3 v) {\n"
            "    vec3 delta = viewPosition - center;\n"
            "    float axial = dot(delta, cycleAxis);\n"
            "    vec3 radial = delta - cycleAxis * axial;\n"
            "    radial /= max(length(radial), 0.0001);\n"
            "    float side = step(0.0, axial) * 2.0 - 1.0;\n"
            "    vec3 source = center + radial * ringShape.x + cycleAxis * side * ringShape.y;\n"
            "    return neonSample(source, n, v, ringShape.z);\n"
            "}\n"
            "vec3 reactorLight(vec3 n, vec3 v) {\n"
            "    vec3 axis = reactorB - reactorA;\n"
            "    float t = clamp(dot(viewPosition - reactorA, axis) / max(dot(axis, axis), 0.000001), 0.0, 1.0);\n"
            "    vec3 center = reactorA + axis * t;\n"
            "    float side = step(0.0, dot(viewPosition - center, cycleAxis)) * 2.0 - 1.0;\n"
            "    return neonSample(center + cycleAxis * side * ringShape.y * 0.96, n, v, ringShape.z * 0.8);\n"
            "}\n"
            "void main() {\n"
            "    vec3 n = normalize(viewNormal);\n"
            "    vec3 v = normalize(-viewPosition);\n"
            "    if (!gl_FrontFacing) n = -n;\n"
            "    vec3 key = normalize(gl_LightSource[0].position.xyz);\n"
            "    vec3 fill = normalize(gl_LightSource[1].position.xyz);\n"
            "    float power = max(gl_FrontMaterial.shininess, 2.0);\n"
            "    float nv = max(dot(n, v), 0.0);\n"
            "    float fresnel = 0.04 + 0.96 * pow(1.0 - nv, 5.0);\n"
            "    vec3 keyColour = vec3(0.72, 0.86, 1.0);\n"
            "    vec3 fillColour = vec3(0.25, 0.48, 0.56);\n"
            "    vec3 illumination = vec3(0.12, 0.18, 0.22);\n"
            "    illumination += keyColour * max(dot(n, key), 0.0) * 1.25;\n"
            "    illumination += fillColour * max(dot(n, fill), 0.0) * 0.75;\n"
            "    vec3 body = gl_FrontMaterial.diffuse.rgb * illumination;\n"
            "    body += gl_FrontMaterial.ambient.rgb * 0.45;\n"
            "    vec3 specular = keyColour * highlight(n, v, key, power);\n"
            "    specular += fillColour * highlight(n, v, fill, power * 0.65) * 0.55;\n"
            // Broad cool reflections reveal curvature even between direct highlights.
            "    vec3 reflected = reflect(-v, n);\n"
            "    float horizon = smoothstep(-0.35, 0.8, reflected.y);\n"
            "    float strip = exp(-pow((reflected.x + 0.20) / 0.13, 2.0));\n"
            "    strip *= smoothstep(-0.4, 0.35, reflected.y);\n"
            "    vec3 environment = mix(vec3(0.012, 0.022, 0.03), vec3(0.15, 0.27, 0.33), horizon);\n"
            "    environment += vec3(0.30, 0.48, 0.55) * strip;\n"
            "    body += gl_FrontMaterial.specular.rgb * (specular * 0.72 + environment * (0.32 + fresnel));\n"
            "    body += ringLight(rearLight, n, v) + ringLight(frontLight, n, v) + reactorLight(n, v);\n"
            "    body = min(body, vec3(0.94));\n"
            "    gl_FragColor = vec4(body + gl_FrontMaterial.emission.rgb, gl_FrontMaterial.diffuse.a);\n"
            "}\n";
        GLuint vs = Compile(GL_VERTEX_SHADER, vertex);
        GLuint fs = Compile(GL_FRAGMENT_SHADER, fragment);
        if (vs && fs)
        {
            Program() = glCreateProgram();
            glAttachShader(Program(), vs);
            glAttachShader(Program(), fs);
            glLinkProgram(Program());
            GLint success = GL_FALSE;
            glGetProgramiv(Program(), GL_LINK_STATUS, &success);
            if (!success)
            {
                glDeleteProgram(Program());
                Program() = 0;
            }
            else
            {
                Lights().colour=glGetUniformLocation(Program(),"neonColour");
                Lights().axis=glGetUniformLocation(Program(),"cycleAxis");
                Lights().shape=glGetUniformLocation(Program(),"ringShape");
                Lights().rear=glGetUniformLocation(Program(),"rearLight");
                Lights().front=glGetUniformLocation(Program(),"frontLight");
                Lights().reactorA=glGetUniformLocation(Program(),"reactorA");
                Lights().reactorB=glGetUniformLocation(Program(),"reactorB");
            }
        }
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        if (!Program())
        {
            Failed() = true;
            con << "Lightcycle material: using compatibility lighting.\n";
        }
        return Program() != 0;
    }
#endif

public:
    class Scope
    {
    public:
        Scope(float r, float g, float b, float rearAxle, float frontAxle, float wheelRadius)
        {
#ifdef HAVE_GLEW
            if (Prepare())
            {
                glGetIntegerv(GL_CURRENT_PROGRAM, &previous_);
                glUseProgram(Program());
                active_ = true;
                GLfloat matrix[16];
                glGetFloatv(GL_MODELVIEW_MATRIX,matrix);
                auto point = [&](GLint uniform,float x,float y,float z)
                {
                    glUniform3f(uniform,
                        matrix[0]*x+matrix[4]*y+matrix[8]*z+matrix[12],
                        matrix[1]*x+matrix[5]*y+matrix[9]*z+matrix[13],
                        matrix[2]*x+matrix[6]*y+matrix[10]*z+matrix[14]);
                };
                float scale=std::sqrt(matrix[4]*matrix[4]+matrix[5]*matrix[5]+matrix[6]*matrix[6]);
                scale=std::max(scale,.0001f);
                glUniform3f(Lights().colour,r,g,b);
                glUniform3f(Lights().axis,matrix[4]/scale,matrix[5]/scale,matrix[6]/scale);
                glUniform3f(Lights().shape,wheelRadius*.84306f*scale,wheelRadius*.61944f*scale,wheelRadius*.50f*scale);
                point(Lights().rear,rearAxle,0,wheelRadius);
                point(Lights().front,frontAxle,0,wheelRadius);
                point(Lights().reactorA,-.32f,0,.73f);
                point(Lights().reactorB,.22f,0,.73f);
            }
#endif
        }
        ~Scope()
        {
#ifdef HAVE_GLEW
            if (active_) glUseProgram(previous_);
#endif
        }
        Scope(Scope const &) = delete;
        Scope & operator=(Scope const &) = delete;
    private:
#ifdef HAVE_GLEW
        GLint previous_ = 0;
        bool active_ = false;
#endif
    };
};
#endif
#endif
