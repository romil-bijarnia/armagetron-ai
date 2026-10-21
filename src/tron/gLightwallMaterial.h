#ifndef ARMAGETRON_LIGHTWALL_MATERIAL_H
#define ARMAGETRON_LIGHTWALL_MATERIAL_H

#include "aa_config.h"

#ifndef DEDICATED
#include "rGL.h"
#include "rRender.h"
#include "rScreen.h"
#include "rDisplayList.h"
#include "rSceneSample.h"
#include "tConsole.h"
#include "gLightColour.h"
#include <algorithm>
#include <cmath>

// Optical membranes over the physical wall mesh. Broad persistent folds bend
// the real scene between luminous borders. Depth rejection preserves nearby
// silhouettes; derivative filtering keeps distant borders stable.
class gLightwallMaterial
{
    static bool & ActiveState() { static bool value=false; return value; }
#ifdef HAVE_GLEW
    struct Bindings
    {
        GLint texture=-1, time=-1, head=-1, viewport=-1;
        GLint scene=-1,depth=-1,captured=-1,depthReady=-1,inverseFrame=-1,bounds=-1,pixelScale=-1;
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
        char const * sources[]={gLightColour::ShaderSource(),source};
        glShaderSource(shader,kind==GL_FRAGMENT_SHADER ? 2 : 1,
                       kind==GL_FRAGMENT_SHADER ? sources : &source,NULL);
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
        char const * vertex=R"GLSL(#version 120
varying vec3 wallCoord;
varying vec4 wallColour;
varying vec4 wallBottomClip;
varying vec4 wallTopClip;
varying vec3 wallEye;
varying vec3 wallUp;
varying float wallMirror;
void main() {
    wallCoord=gl_MultiTexCoord0.xyz;
    wallColour=gl_Color;
    float height=max(gl_MultiTexCoord0.w,0.0001);
    wallBottomClip=gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xy,0.0,1.0);
    wallTopClip=gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xy,height,1.0);
    gl_ClipVertex=gl_ModelViewMatrix*gl_Vertex;
    wallEye=gl_ClipVertex.xyz;
    wallUp=(gl_ModelViewMatrix*vec4(0.0,0.0,1.0,0.0)).xyz;
    wallMirror=1.0-step(0.0,dot(cross(gl_ModelViewMatrix[0].xyz,gl_ModelViewMatrix[1].xyz),gl_ModelViewMatrix[2].xyz));
    gl_Position=ftransform();
}
)GLSL";
        char const * fragment=R"GLSL(
uniform sampler2D energyTexture;
uniform sampler2D sceneTexture;
uniform sampler2D sceneDepth;
uniform float sceneCaptured;
uniform float depthCaptured;
uniform float flowTime;
uniform float headCoord;
uniform float pixelScale;
uniform vec2 referenceViewport;
uniform vec2 inverseFrame;
uniform vec4 sceneBounds;
varying vec3 wallCoord;
varying vec4 wallColour;
varying vec4 wallBottomClip;
varying vec4 wallTopClip;
varying vec3 wallEye;
varying vec3 wallUp;
varying float wallMirror;
float edgeCoverage(float v,float rate,float footprint) {
    float lo=max(0.0,v-footprint*0.5),hi=min(1.0,v+footprint*0.5);
    return (exp(-rate*lo)-exp(-rate*hi))/(rate*footprint);
}
float hash21(vec2 p) {
    vec3 q=fract(vec3(p.xyx)*vec3(0.1031,0.1030,0.0973));
    q+=dot(q,q.yzx+33.33);
    return fract((q.x+q.y)*q.z);
}
float opticalNoise(vec2 p) {
    vec2 i=floor(p),f=fract(p);
    f=f*f*(3.0-2.0*f);
    return mix(mix(hash21(i),hash21(i+vec2(1.0,0.0)),f.x),
               mix(hash21(i+vec2(0.0,1.0)),hash21(i+vec2(1.0)),f.x),f.y);
}
float contour(float field,float level,float width) {
    float aa=max(fwidth(field),0.001);
    float filtered=sqrt(width*width+aa*aa);
    return exp(-pow((field-level)/filtered,2.0))*width/filtered;
}
// Reflect visible arena geometry in the upright sheet. A bounded ray walk uses
// the same opaque colour/depth snapshot as refraction. Missing/off-screen hits
// contribute nothing; they never fetch another viewport or invent a silhouette.
vec4 reflection(vec3 origin,vec3 direction) {
    float travel=0.12;
    for (int i=0;i<16;++i) {
        float stride=0.13+float(i)*0.085;
        travel+=stride;
        vec3 at=origin+direction*travel;
        vec4 projected=gl_ProjectionMatrix*vec4(at,1.0);
        if (projected.w<=0.01) break;
        vec2 uv=projected.xy/projected.w*0.5+0.5;
        if (min(uv.x,uv.y)<0.002 || max(uv.x,uv.y)>0.998) break;
        uv=mix(sceneBounds.xy,sceneBounds.zw,uv);
        float depth=texture2D(sceneDepth,uv).r;
        float sceneDistance=gl_ProjectionMatrix[3][2]/(depth*2.0-1.0+gl_ProjectionMatrix[2][2]);
        float delta=-at.z-sceneDistance;
        if (depth<0.99999 && delta>0.0 && delta<stride*1.4) {
            float edge=min(min(uv.x-sceneBounds.x,sceneBounds.z-uv.x),
                           min(uv.y-sceneBounds.y,sceneBounds.w-uv.y));
            return vec4(texture2D(sceneTexture,uv).rgb,smoothstep(0.0,0.035,edge));
        }
    }
    return vec4(0.0);
}
vec4 distantSurface(float u,float v,float footprint,float pixels,vec2 uv) {
    vec3 tint=wallColour.rgb;
    vec3 core=lightCore(tint,0.25);
    float width=max(0.028,0.35/max(pixels,1.0));
    float upper=edgeCoverage(v,1.0/width,footprint);
    float lower=edgeCoverage(1.0-v,0.78/width,footprint);
    float shoulders=edgeCoverage(v,8.0,footprint)+edgeCoverage(1.0-v,5.5,footprint);
    float flow=0.016*sin(u*0.42-flowTime*0.45);
    // Minified smoked glass retains a dark centre and a broad luminous foot.
    // A uniform coloured fill made separated trails read as solid plastic bars.
    vec3 light=tint*(0.055+0.13*pow(v,3.0)+flow+0.10*shoulders);
    // A distant mirror image is a soft pool under the sheet, not another pair
    // of equally sharp rails. This only affects the reflected, minified trail.
    light+=core*(upper*0.85*mix(1.0,0.35,wallMirror)+lower*1.10);
    light*=mix(1.0,0.68,wallMirror);
    float alpha=wallColour.a*0.28;
    vec3 scene=texture2D(sceneTexture,clamp(uv,sceneBounds.xy,sceneBounds.zw)).rgb;
    return vec4(scene*(vec3(0.82)+tint*0.18)*alpha*sceneCaptured+boundedLight(light,tint)*wallColour.a,alpha);
}
void main() {
    vec2 bottom=wallBottomClip.xy/max(wallBottomClip.w,0.0001);
    vec2 top=wallTopClip.xy/max(wallTopClip.w,0.0001);
    vec2 vertical=(top-bottom)*referenceViewport;
    float ribbonPixels=length(vertical);
    vertical/=max(ribbonPixels,0.001);
    float detail=smoothstep(1.0,5.0,ribbonPixels);
    if (wallCoord.z < -0.5) {
        if (wallCoord.z < -1.5) {
            vec4 spill=wallColour*texture2D(energyTexture,wallCoord.xy);
            gl_FragColor=vec4(spill.rgb*spill.a,0.0);
        } else discard;
        return;
    }
    float u=wallCoord.x,v=clamp(wallCoord.y,0.0,1.0);
    float footprint=max(fwidth(wallCoord.y),0.0001);
    // The top-to-bottom vector alone overestimates visible thickness when a
    // wall is seen end-on. The UV gradient measures its perpendicular footprint
    // and is normalized to the same 1080p reference as the close material.
    float sheetPixels=1.0/max(length(vec2(dFdx(wallCoord.y),dFdy(wallCoord.y)))*pixelScale,0.0001);
    float closeDetail=smoothstep(10.0,32.0,min(ribbonPixels,sheetPixels));
    vec2 uv=gl_FragCoord.xy*inverseFrame;
    if (closeDetail<=0.0) {
        gl_FragColor=distantSurface(u,v,footprint,min(ribbonPixels,sheetPixels),uv);
        return;
    }
    // Preserve the optical core at normal playing distances. Its emission is
    // area-filtered once, independently of the transparency of the glass.
    float coreWidth=max(0.028,0.85/max(ribbonPixels,1.0));
    float upper=edgeCoverage(v,1.0/coreWidth,footprint);
    float lower=edgeCoverage(1.0-v,0.78/coreWidth,footprint);
    float shoulder=edgeCoverage(v,8.0,footprint)+edgeCoverage(1.0-v,5.5,footprint);
    float border=upper+lower;
    float inside=smoothstep(0.012,0.10,v)*(1.0-smoothstep(0.90,0.988,v));
    float behind=max(0.0,headCoord-u);
    float newborn=exp(-behind*0.65);

    // Advection through two broad distortion fields creates stretched lenses
    // and changing caustic contours, rather than a repeated sine-wave stripe.
    // Distance UVs continue through segment boundaries and retained wall lists.
    float t=flowTime;
    vec2 p=vec2(u*0.46-t*0.38,v*3.1);
    vec2 drift=vec2(opticalNoise(p*0.72+vec2(0.0,t*0.12)),
                    opticalNoise(p*0.86+vec2(9.7,-t*0.16)))-0.5;
    vec2 lens=p+drift*1.65;
    float field=opticalNoise(lens);
    float fine=opticalNoise(lens*2.15+vec2(7.2,t*0.13));
    float fold=contour(field,0.48,0.065);
    float caustic=contour(field+fine*0.15,0.56,0.018);
    float sweep=sin(u*0.63-t*0.84+v*4.0+drift.x*2.8);
    float streak=contour(v+drift.y*0.42,0.52+0.12*sweep,0.025);
    // Fine fibres follow the same moving optical field, so their direction
    // agrees with the broad folds. Derivative filtering removes unresolved
    // fibres; the emission wake is confined to the newly laid sheet.
    float shear=field+fine*0.12;
    float fibres=contour(shear,0.31,0.009)+0.65*contour(shear,0.73,0.007);
    float wake=exp(-behind*0.30);
    float wakeLine=contour(v+drift.y*0.30,0.5+0.16*sin(behind*1.8-t*1.4),0.022);
    float motion=inside*detail;
    vec2 along=vec2(vertical.y,-vertical.x);
    vec2 warp=along*(drift.x*17.0+sweep*2.0)
             +vertical*((field-0.5)*0.48*ribbonPixels+drift.y*8.0);
    warp*=motion*pixelScale*closeDetail;
    vec2 warped=clamp(uv+warp*inverseFrame,sceneBounds.xy,sceneBounds.zw);
    if (depthCaptured>0.5 && texture2D(sceneDepth,warped).r<gl_FragCoord.z-0.00002)
        warped=clamp(uv,sceneBounds.xy,sceneBounds.zw);
    vec3 scenery=texture2D(sceneTexture,warped).rgb;
    vec3 tint=wallColour.rgb;
    float cool=cyanLight(tint);
    vec3 hot=lightCore(tint,1.0);
    vec3 normal=normalize(cross(dFdx(wallEye),dFdy(wallEye)));
    vec3 view=normalize(-wallEye);
    if (dot(normal,view)<0.0) normal=-normal;
    vec3 tangent=normalize(cross(wallUp,normal));
    vec3 bitangent=normalize(cross(normal,tangent));
    vec3 opticalNormal=normalize(normal+tangent*drift.x*0.12+bitangent*drift.y*0.20);
    vec4 reflected=vec4(0.0);
    if (depthCaptured>0.5 && detail>0.1 && closeDetail>0.1)
        reflected=reflection(wallEye+normal*0.035,reflect(-view,opticalNormal));
    float fresnel=0.24+0.50*pow(1.0-abs(dot(view,opticalNormal)),3.0);
    // Absorption governs the view through the sheet; emitted light must not be
    // multiplied by that absorption a second time. This was making thin walls
    // and their reflections disappear from the ordinary elevated camera.
    float opacity=clamp(0.20+inside*0.16+fold*motion*0.12,0.0,0.60);
    float alpha=wallColour.a*opacity;
    vec3 glass=scenery*(vec3(0.82)+tint*0.18);
    vec3 transmitted=glass*alpha*sceneCaptured;
    vec3 volume=tint*(0.055+0.07*field+motion*(0.16*fold+0.12*caustic));
    // Dark optical pockets between the folds are visible in the 4K reference.
    // Modulate only the internal scatter; keep transmission and rim light intact.
    volume*=mix(0.52,1.0,smoothstep(0.26,0.48,field));
    volume+=hot*motion*(0.12*caustic+0.045*streak);
    volume+=mix(tint,hot,0.45)*motion*(0.065*fibres+0.12*wake*wakeLine);
    volume+=reflected.rgb*reflected.a*fresnel*motion*(vec3(0.65)+tint*0.35);
    vec3 emission=hot*(upper*mix(1.10,1.70,cool)+lower*mix(1.35,2.10,cool))+tint*shoulder*0.42;
    emission+=volume+hot*newborn*inside*0.065;
    emission=boundedLight(emission,tint);
    vec4 closeColour=vec4(transmitted+emission*wallColour.a,alpha);
    // Subpixel glass needs an integrated light field, not two white rails plus
    // high-frequency scene fetches. Preserve its hue and broad travelling flow.
    // At 32 reference pixels and above, use the full liquid-glass field.
    if (closeDetail>=1.0) gl_FragColor=closeColour;
    else gl_FragColor=mix(distantSurface(u,v,footprint,min(ribbonPixels,sheetPixels),uv),closeColour,closeDetail);
}
)GLSL";
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
        if (sr_hiddenWindow) std::cerr << "Legacy wall material: scene distortion shader linked.\n";
        Uniforms().texture=glGetUniformLocation(Program(),"energyTexture");
        Uniforms().time=glGetUniformLocation(Program(),"flowTime");
        Uniforms().head=glGetUniformLocation(Program(),"headCoord");
        Uniforms().viewport=glGetUniformLocation(Program(),"referenceViewport");
        Uniforms().scene=glGetUniformLocation(Program(),"sceneTexture");
        Uniforms().depth=glGetUniformLocation(Program(),"sceneDepth");
        Uniforms().captured=glGetUniformLocation(Program(),"sceneCaptured");
        Uniforms().depthReady=glGetUniformLocation(Program(),"depthCaptured");
        Uniforms().inverseFrame=glGetUniformLocation(Program(),"inverseFrame");
        Uniforms().bounds=glGetUniformLocation(Program(),"sceneBounds");
        Uniforms().pixelScale=glGetUniformLocation(Program(),"pixelScale");
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
            rSceneSample::State const & scene=rSceneSample::Get();
            glUniform1i(Uniforms().scene,1);
            glUniform1i(Uniforms().depth,2);
            glUniform1f(Uniforms().captured,scene.ready ? 1.f : 0.f);
            glUniform1f(Uniforms().depthReady,scene.depthReady ? 1.f : 0.f);
            glUniform2f(Uniforms().inverseFrame,1.f/std::max(1,sr_screenWidth),1.f/std::max(1,sr_screenHeight));
            glUniform1f(Uniforms().pixelScale,sr_screenHeight/1080.f);
            glUniform4f(Uniforms().bounds,(scene.viewport[0]+.5f)/std::max(1,sr_screenWidth),
                        (scene.viewport[1]+.5f)/std::max(1,sr_screenHeight),
                        (scene.viewport[0]+scene.viewport[2]-.5f)/std::max(1,sr_screenWidth),
                        (scene.viewport[1]+scene.viewport[3]-.5f)/std::max(1,sr_screenHeight));
            for (int unit=1;unit<=2;++unit)
            {
                glActiveTexture(GL_TEXTURE0+unit);
                glGetIntegerv(GL_TEXTURE_BINDING_2D,&previousScene_[unit-1]);
                glBindTexture(GL_TEXTURE_2D,unit==1 ? scene.colour : scene.depth);
            }
            sceneBound_=true;
            glActiveTexture(GL_TEXTURE0);
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT,viewport);
            // Measure apparent height in 1080p reference pixels. Raising native
            // resolution improves sampling without restoring distant rail bars.
            glUniform2f(Uniforms().viewport,540.f*viewport[2]/std::max(1,viewport[3]),540.f);
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
                if (sceneBound_)
                    for (int unit=1;unit<=2;++unit)
                    {
                        glActiveTexture(GL_TEXTURE0+unit);
                        glBindTexture(GL_TEXTURE_2D,previousScene_[unit-1]);
                    }
                glActiveTexture(previousTexture_);
            }
#endif
        }
        Scope(Scope const &)=delete;
        Scope & operator=(Scope const &)=delete;
    private:
#ifdef HAVE_GLEW
        GLint previousProgram_=0, previousTexture_=GL_TEXTURE0;
        GLint previousScene_[2]={0,0};
        bool sceneBound_=false;
        bool bound_=false,previousActive_=false;
#endif
    };
};
#endif
#endif
