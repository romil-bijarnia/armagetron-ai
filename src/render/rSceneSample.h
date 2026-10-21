// A per-view snapshot for transparent optical materials. GPL-2.0-or-later.
#ifndef ARMAGETRON_SCENE_SAMPLE_H
#define ARMAGETRON_SCENE_SAMPLE_H

#include "aa_config.h"
#ifndef DEDICATED
#include "rGL.h"
#include "rRender.h"
#include "rScreen.h"
#include "tConsole.h"

namespace rSceneSample
{
#ifdef HAVE_GLEW
struct State
{
    GLuint colour=0, depth=0;
    GLint width=0,height=0,viewport[4]={0,0,0,0};
    bool ready=false,depthReady=false,failed=false,reported=false,validated=false;
};
inline State & Get() { static State state; return state; }
inline void Release()
{
    State & s=Get();
    if (s.colour) glDeleteTextures(1,&s.colour);
    if (s.depth) glDeleteTextures(1,&s.depth);
    s=State();
}
inline void BeginView() { Get().ready=Get().depthReady=false; }
inline void Capture()
{
    static rCallbackBeforeScreenModeChange cleanup(&Release);
    State & s=Get();
    s.ready=s.depthReady=false;
    if (s.failed || !GLEW_VERSION_2_0 || sr_screenWidth<=0 || sr_screenHeight<=0) return;
    RenderEnd();
    GLint active=GL_TEXTURE0,binding=0;
    glGetIntegerv(GL_ACTIVE_TEXTURE,&active);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D,&binding);
    glGetIntegerv(GL_VIEWPORT,s.viewport);
    GLenum prior=glGetError();
    if (prior!=GL_NO_ERROR && !s.reported)
    {
        std::cerr << "Scene sample: pre-existing render error " << int(prior) << "\n";
        s.reported=true;
    }
    if (!s.colour) glGenTextures(1,&s.colour);
    if (!s.depth) glGenTextures(1,&s.depth);
    bool resize=s.width!=sr_screenWidth || s.height!=sr_screenHeight;
    for (int pass=0;pass<2;++pass)
    {
        glBindTexture(GL_TEXTURE_2D,pass ? s.depth : s.colour);
        if (resize)
        {
            glTexImage2D(GL_TEXTURE_2D,0,pass ? GL_DEPTH_COMPONENT24 : GL_RGB8,
                         sr_screenWidth,sr_screenHeight,0,pass ? GL_DEPTH_COMPONENT : GL_RGB,
                         pass ? GL_UNSIGNED_INT : GL_UNSIGNED_BYTE,NULL);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,pass ? GL_NEAREST : GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,pass ? GL_NEAREST : GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            if (pass) glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_COMPARE_MODE,GL_NONE);
        }
        // Copy only this camera's viewport. Sampling is clamped to these bounds;
        // the mirrored scene and split-screen cameras never reuse stale pixels.
        glCopyTexSubImage2D(GL_TEXTURE_2D,0,s.viewport[0],s.viewport[1],
                           s.viewport[0],s.viewport[1],s.viewport[2],s.viewport[3]);
        GLenum error=glGetError();
        if (pass) s.depthReady=error==GL_NO_ERROR;
        else s.ready=error==GL_NO_ERROR;
        if (error!=GL_NO_ERROR && !s.reported)
        {
            std::cerr << "Scene sample: " << (pass ? "depth" : "colour") << " capture error " << int(error) << "\n";
            s.reported=true;
        }
    }
    if (sr_hiddenWindow && s.ready && !s.validated)
    {
        std::cerr << "Optical scene capture: colour=" << s.ready << " depth=" << s.depthReady
                  << " size=" << sr_screenWidth << "x" << sr_screenHeight << '\n';
        s.validated=true;
    }
    s.width=sr_screenWidth;s.height=sr_screenHeight;
    s.failed=!s.ready;
    glBindTexture(GL_TEXTURE_2D,binding);
    glActiveTexture(active);
}
#else
inline void BeginView() {}
inline void Capture() {}
#endif
}
#endif
#endif
