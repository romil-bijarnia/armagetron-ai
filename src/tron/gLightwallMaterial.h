#ifndef ARMAGETRON_LIGHTWALL_MATERIAL_H
#define ARMAGETRON_LIGHTWALL_MATERIAL_H

#include "aa_config.h"

#ifndef DEDICATED
#include "rGL.h"
#include "rRender.h"
#include "rScreen.h"
#include "rDisplayList.h"
#include "tConsole.h"
#include <algorithm>
#include <cmath>

// A material over the existing wall mesh. Only recent portions of the energy
// sheet carry strong moving caustics; collision filaments remain crisp and the
// older wall stays quiet. The RGBA texture is the compatibility fallback.
class gLightwallMaterial
{
    static bool & ActiveState() { static bool value=false; return value; }
#ifdef HAVE_GLEW
    struct Bindings
    {
        GLint texture=-1, time=-1, head=-1;
    };
    static GLuint & Program() { static GLuint value=0; return value; }
    static bool & Failed() { static bool value=false; return value; }
    static Bindings & Uniforms() { static Bindings value; return value; }
    static void Unload()
    {
        if (Program()) glDeleteProgram(Program());
        Program()=0;
        Failed()=false;
        Uniforms()=Bindings();
        ActiveState()=false;
    }
    static GLuint Compile(GLenum kind,char const * source)
    {
        GLuint shader=glCreateShader(kind);
        glShaderSource(shader,1,&source,NULL);
        glCompileShader(shader);
        GLint success=GL_FALSE;
        glGetShaderiv(shader,GL_COMPILE_STATUS,&success);
        if (!success)
        {
            char message[1024]="";
            glGetShaderInfoLog(shader,sizeof(message)-1,NULL,message);
            con << "Lightwall material: " << message << "\n";
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
        char const * vertex=
            "#version 120\n"
            "varying vec3 wallCoord;\n"
            "varying vec4 wallColour;\n"
            "void main() {\n"
            "    wallCoord = gl_MultiTexCoord0.xyz;\n"
            "    wallColour = gl_Color;\n"
            "    gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n"
            "    gl_Position = ftransform();\n"
            "}\n";
        char const * fragment=
            "#version 120\n"
            "uniform sampler2D energyTexture;\n"
            "uniform float flowTime;\n"
            "uniform float headCoord;\n"
            "varying vec3 wallCoord;\n"
            "varying vec4 wallColour;\n"
            "float filamentMask(float field, float coreWidth) {\n"
            "    float width = clamp(fwidth(field) * 0.75, 0.012, 0.18);\n"
            "    return 1.0 - smoothstep(coreWidth, coreWidth + width, abs(field));\n"
            "}\n"
            "void main() {\n"
            // Negative third coordinates are stable material markers baked in
            // the mesh, never per-draw uniform changes captured in a wall list.
            "    if (wallCoord.z < -0.5) {\n"
            "        if (wallCoord.z < -1.5) {\n"
            "            vec4 spill = wallColour * texture2D(energyTexture, wallCoord.xy);\n"
            "            gl_FragColor = vec4(spill.rgb * spill.a, 0.0);\n"
            "        } else gl_FragColor = vec4(wallColour.rgb * wallColour.a, wallColour.a);\n"
            "        return;\n"
            "    }\n"
            "    vec4 sheet = texture2D(energyTexture, wallCoord.xy);\n"
            "    float u = wallCoord.x;\n"
            "    float v = clamp(wallCoord.y, 0.0, 1.0);\n"
            "    float behind = max(0.0, headCoord - u);\n"
            "    if (behind >= 12.0) {\n"
            "        float alpha = clamp(wallColour.a * sheet.a, 0.0, 1.0);\n"
            "        gl_FragColor = vec4(wallColour.rgb * sheet.rgb * alpha, alpha);\n"
            "        return;\n"
            "    }\n"
            "    float recent = exp(-behind * 0.38);\n"
            "    float strength = (0.035 + 0.965 * recent) * (1.0 - smoothstep(8.0, 12.0, behind));\n"
            "    float interior = smoothstep(0.018, 0.10, v) * (1.0 - smoothstep(0.90, 0.982, v));\n"
            "    vec2 q = vec2(u * 1.04 - flowTime * 0.12, v * 2.275);\n"
            "    float wx = 0.46 * sin(v * 5.2 + sin(u * 1.105 + flowTime * 0.27));\n"
            "    float wy = 0.18 * sin(u * 2.925 - flowTime * 0.38);\n"
            "    wy += 0.10 * sin(u * 5.265 + v * 3.51 + flowTime * 0.21);\n"
            "    q += vec2(wx, wy);\n"
            "    float network = sin(q.x * 5.1 + q.y * 1.7);\n"
            "    network += 0.75 * sin(q.y * 5.8 - q.x * 0.9);\n"
            "    network += 0.28 * sin(q.x * 9.1 + q.y * 3.4);\n"
            "    float crossing = sin(q.x * 6.2 - q.y * 4.0 + flowTime * 0.25);\n"
            "    crossing += 0.70 * sin(q.y * 6.6 + q.x * 1.5 - flowTime * 0.15);\n"
            "    crossing += 0.22 * network;\n"
            // Derivative filtering prevents subpixel filaments from crawling;
            // the wider shoulder remains a subtle ripple, not an opaque fill.
            "    float ripple = 0.5 + 0.5 * sin(u * 2.6 - v * 5.85 + flowTime * 0.50 + wx * 2.0);\n"
            "    float first = filamentMask(network, 0.040 + 0.040 * ripple) * (0.58 + 0.42 * ripple);\n"
            "    float second = filamentMask(crossing, 0.025 + 0.035 * (1.0 - ripple)) * (0.35 + 0.45 * (1.0 - ripple));\n"
            "    float shoulder = exp(-min(abs(network), abs(crossing)) * 2.8);\n"
            "    float energy = clamp(interior * strength * (first + second + 0.24 * shoulder + 0.04 * ripple * recent), 0.0, 1.0);\n"
            "    vec3 brightness = min(vec3(1.0), sheet.rgb * (1.0 + 0.035 * (ripple - 0.5) * strength) + 0.70 * energy);\n"
            "    float opacity = clamp(sheet.a + 0.65 * energy + 0.025 * interior * ripple * strength, 0.0, 1.0);\n"
            "    float alpha = clamp(wallColour.a * opacity, 0.0, 1.0);\n"
            "    gl_FragColor = vec4(wallColour.rgb * brightness * alpha, alpha);\n"
            "}\n";
        GLuint vs=Compile(GL_VERTEX_SHADER,vertex), fs=Compile(GL_FRAGMENT_SHADER,fragment);
        if (vs && fs)
        {
            Program()=glCreateProgram();
            glAttachShader(Program(),vs);
            glAttachShader(Program(),fs);
            glLinkProgram(Program());
            GLint linked=GL_FALSE;
            glGetProgramiv(Program(),GL_LINK_STATUS,&linked);
            if (!linked)
            {
                char message[1024]="";
                glGetProgramInfoLog(Program(),sizeof(message)-1,NULL,message);
                con << "Lightwall material link: " << message << "\n";
                glDeleteProgram(Program());
                Program()=0;
            }
        }
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        if (!Program())
        {
            Failed()=true;
            con << "Lightwall material: using the textured energy sheet.\n";
            return false;
        }
        Uniforms().texture=glGetUniformLocation(Program(),"energyTexture");
        Uniforms().time=glGetUniformLocation(Program(),"flowTime");
        Uniforms().head=glGetUniformLocation(Program(),"headCoord");
        return true;
    }
#endif
public:
    // Wall lists record the matching blend mode; this is true only after a
    // program has actually been bound, not merely when shaders are supported.
    static bool IsActive() { return ActiveState(); }

    class Scope
    {
    public:
        Scope(bool enabled,float seconds,float head)
        {
#ifdef HAVE_GLEW
            // The caller enters before creating/calling any wall display list.
            // A surrounding renderer-owned list also falls back rather than
            // baking a program switch or stale animation uniforms into itself.
            if (!enabled || rDisplayList::IsRecording() || !GLEW_VERSION_2_0) return;
            RenderEnd();
            glGetIntegerv(GL_CURRENT_PROGRAM,&previousProgram_);
            glGetIntegerv(GL_ACTIVE_TEXTURE,&previousTexture_);
            previousActive_=ActiveState();
            glActiveTexture(GL_TEXTURE0);
            bound_=true;
            if (!Prepare())
            {
                // A failed optional shader must leave a real fixed-function
                // fallback, rather than inheriting another material's program.
                glUseProgram(0);
                ActiveState()=false;
                return;
            }
            glUseProgram(Program());
            float clock=std::isfinite(seconds) ? std::fmod(std::max(0.f,seconds),2048.f) : 0;
            float distance=std::isfinite(head) ? std::max(0.f,head) : 0;
            glUniform1i(Uniforms().texture,0);
            glUniform1f(Uniforms().time,clock);
            glUniform1f(Uniforms().head,distance);
            ActiveState()=true;
#else
            (void)enabled; (void)seconds; (void)head;
#endif
        }
        ~Scope()
        {
#ifdef HAVE_GLEW
            if (bound_)
            {
                RenderEnd();
                ActiveState()=previousActive_;
                glUseProgram(previousProgram_);
                glActiveTexture(previousTexture_);
            }
#endif
        }
        Scope(Scope const &)=delete;
        Scope & operator=(Scope const &)=delete;
    private:
#ifdef HAVE_GLEW
        GLint previousProgram_=0, previousTexture_=GL_TEXTURE0;
        bool bound_=false,previousActive_=false;
#endif
    };
};
#endif
#endif
