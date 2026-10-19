/*

*************************************************************************

ArmageTron -- Just another Tron Lightcycle Game in 3D.
Copyright (C) 2026, Armagetron Advanced Development Team

**************************************************************************

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

***************************************************************************

*/

#include "rBloom.h"

#include "aa_config.h"

#ifndef DEDICATED

// this file issues raw OpenGL calls on purpose
#define DONTDOIT
#include "rRender.h"
#include "rGL.h"
#include "rScreen.h"
#include "rTextureRenderTarget.h"
#include "tConfiguration.h"
#include "tConsole.h"

// The glow is a single-level bloom: the finished scene is copied into a texture,
// its bright parts are extracted into a glow buffer a quarter (an eighth on tall
// screens) of the screen size, blurred with a separable gaussian and added back
// onto the scene. Settings:
//   BLOOM            on or off
//   BLOOM_INTENSITY  how much glow is added (0 disables)
//   BLOOM_THRESHOLD  brightness (max of r, g, b) from which a pixel glows
//   BLOOM_RADIUS     tap spacing of the blur in glow buffer pixels; wider halo for more
static bool sr_bloom = false;
static REAL sr_bloomIntensity = 2.5;
static REAL sr_bloomThreshold = .5;
static REAL sr_bloomRadius = 2;

static tSettingItem< bool > sr_bloomConf( "BLOOM", sr_bloom );
static tSettingItem< REAL > sr_bloomIntensityConf( "BLOOM_INTENSITY", sr_bloomIntensity );
static tSettingItem< REAL > sr_bloomThresholdConf( "BLOOM_THRESHOLD", sr_bloomThreshold );
static tSettingItem< REAL > sr_bloomRadiusConf( "BLOOM_RADIUS", sr_bloomRadius );

#ifdef HAVE_GLEW

namespace
{

//! a colour texture that can be drawn into
struct Target
{
    GLuint texture = 0;
    GLuint framebuffer = 0;
    int width = 0, height = 0;
};

GLuint sceneTexture = 0;         // copy of the finished scene, screen size
int sceneWidth = 0, sceneHeight = 0;
Target glowTarget, scratchTarget; // glow buffer size, ping-pong pair; the glow ends up in glowTarget
GLuint brightProgram = 0, blurProgram = 0;
GLint brightTexel = -1, brightThreshold = -1, brightTaps = -1, blurStep = -1;
bool failed = false;             // after an error the pass stays off until the next screen mode change
bool reported = false;           // the reason is printed once

char const * vertexSource =
    "#version 120\n"
    "void main()\n"
    "{\n"
    "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
    "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
    "}\n";

// taps x taps bilinear taps, two screen pixels apart, cover every screen pixel behind
// one glow pixel; each tap is thresholded on its own so thin bright lines survive
char const * brightSource =
    "#version 120\n"
    "uniform sampler2D scene;\n"
    "uniform vec2 texel;\n"
    "uniform float threshold;\n"
    "uniform int taps;\n"
    "vec3 bright( vec2 uv )\n"
    "{\n"
    "    vec3 c = texture2D( scene, uv ).rgb;\n"
    "    float m = max( c.r, max( c.g, c.b ) );\n"
    "    return c * smoothstep( threshold * .7, threshold, m );\n"
    "}\n"
    "void main()\n"
    "{\n"
    "    vec2 uv = gl_TexCoord[0].xy;\n"
    "    float start = -float( taps - 1 );\n"
    "    vec3 sum = vec3( 0.0 );\n"
    "    for ( int y = 0; y < taps; ++y )\n"
    "        for ( int x = 0; x < taps; ++x )\n"
    "            sum += bright( uv + texel * vec2( start + 2.0 * float( x ), start + 2.0 * float( y ) ) );\n"
    "    gl_FragColor = vec4( sum / float( taps * taps ), 1.0 );\n"
    "}\n";

// 13 tap gaussian (sigma 2.5 taps) along the step direction
char const * blurSource =
    "#version 120\n"
    "uniform sampler2D source;\n"
    "uniform vec2 step;\n"
    "vec3 pair( vec2 uv, float i )\n"
    "{\n"
    "    return texture2D( source, uv + step * i ).rgb + texture2D( source, uv - step * i ).rgb;\n"
    "}\n"
    "void main()\n"
    "{\n"
    "    vec2 uv = gl_TexCoord[0].xy;\n"
    "    vec3 c = texture2D( source, uv ).rgb * .1610;\n"
    "    c += pair( uv, 1.0 ) * .1486;\n"
    "    c += pair( uv, 2.0 ) * .1169;\n"
    "    c += pair( uv, 3.0 ) * .0784;\n"
    "    c += pair( uv, 4.0 ) * .0448;\n"
    "    c += pair( uv, 5.0 ) * .0218;\n"
    "    c += pair( uv, 6.0 ) * .0090;\n"
    "    gl_FragColor = vec4( c, 1.0 );\n"
    "}\n";

void Fail( char const * reason )
{
    failed = true;
    if ( !reported )
    {
        reported = true;
        con << "Bloom: " << reason << ", the glow is off.\n";
    }
}

GLuint Compile( GLenum type, char const * source )
{
    GLuint shader = glCreateShader( type );
    glShaderSource( shader, 1, &source, NULL );
    glCompileShader( shader );
    GLint ok = 0;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &ok );
    if ( !ok )
    {
        char log[1024] = "";
        glGetShaderInfoLog( shader, sizeof( log ) - 1, NULL, log );
        con << "Bloom: shader did not compile: " << log << "\n";
        glDeleteShader( shader );
        return 0;
    }
    return shader;
}

GLuint Link( char const * fragmentSource )
{
    GLuint vertex = Compile( GL_VERTEX_SHADER, vertexSource );
    GLuint fragment = Compile( GL_FRAGMENT_SHADER, fragmentSource );
    GLuint program = 0;
    if ( vertex && fragment )
    {
        program = glCreateProgram();
        glAttachShader( program, vertex );
        glAttachShader( program, fragment );
        glLinkProgram( program );
        GLint ok = 0;
        glGetProgramiv( program, GL_LINK_STATUS, &ok );
        if ( !ok )
        {
            char log[1024] = "";
            glGetProgramInfoLog( program, sizeof( log ) - 1, NULL, log );
            con << "Bloom: shader did not link: " << log << "\n";
            glDeleteProgram( program );
            program = 0;
        }
    }
    // the program keeps the shaders it needs alive
    if ( vertex )
        glDeleteShader( vertex );
    if ( fragment )
        glDeleteShader( fragment );
    return program;
}

void Allocate( GLuint texture, int width, int height )
{
    glBindTexture( GL_TEXTURE_2D, texture );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
}

void Destroy( Target & target )
{
    if ( target.framebuffer )
        glDeleteFramebuffersEXT( 1, &target.framebuffer );
    if ( target.texture )
        glDeleteTextures( 1, &target.texture );
    target = Target();
}

bool Create( Target & target, int width, int height )
{
    Destroy( target );
    glGenTextures( 1, &target.texture );
    Allocate( target.texture, width, height );
    glGenFramebuffersEXT( 1, &target.framebuffer );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, target.framebuffer );
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, target.texture, 0 );
    bool complete = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT ) == GL_FRAMEBUFFER_COMPLETE_EXT;
    target.width = width;
    target.height = height;
    return complete;
}

void Quad()
{
    glBegin( GL_QUADS );
    glTexCoord2f( 0, 0 );
    glVertex2f( -1, -1 );
    glTexCoord2f( 1, 0 );
    glVertex2f( 1, -1 );
    glTexCoord2f( 1, 1 );
    glVertex2f( 1, 1 );
    glTexCoord2f( 0, 1 );
    glVertex2f( -1, 1 );
    glEnd();
}

// frees everything; the next frame builds it again for the new context
void Unload()
{
    if ( brightProgram )
        glDeleteProgram( brightProgram );
    if ( blurProgram )
        glDeleteProgram( blurProgram );
    brightProgram = blurProgram = 0;
    if ( sceneTexture )
        glDeleteTextures( 1, &sceneTexture );
    sceneTexture = 0;
    sceneWidth = sceneHeight = 0;
    Destroy( glowTarget );
    Destroy( scratchTarget );
    failed = false;
}

// the glow buffer is a quarter of the screen, an eighth on tall screens, so the halo
// covers the same share of the picture at every size
int Shrink( int height )
{
    return height >= 1200 ? 8 : 4;
}

bool Prepare( int width, int height )
{
    if ( !brightProgram )
    {
        brightProgram = Link( brightSource );
        if ( !brightProgram )
            return false;
        glUseProgram( brightProgram );
        glUniform1i( glGetUniformLocation( brightProgram, "scene" ), 0 );
        brightTexel = glGetUniformLocation( brightProgram, "texel" );
        brightThreshold = glGetUniformLocation( brightProgram, "threshold" );
        brightTaps = glGetUniformLocation( brightProgram, "taps" );
        glUseProgram( 0 );
    }
    if ( !blurProgram )
    {
        blurProgram = Link( blurSource );
        if ( !blurProgram )
            return false;
        glUseProgram( blurProgram );
        glUniform1i( glGetUniformLocation( blurProgram, "source" ), 0 );
        blurStep = glGetUniformLocation( blurProgram, "step" );
        glUseProgram( 0 );
    }
    if ( sceneWidth != width || sceneHeight != height )
    {
        if ( !sceneTexture )
            glGenTextures( 1, &sceneTexture );
        Allocate( sceneTexture, width, height );
        sceneWidth = width;
        sceneHeight = height;
    }
    int shrink = Shrink( height );
    int glowWidth = ( width + shrink - 1 ) / shrink, glowHeight = ( height + shrink - 1 ) / shrink;
    if ( glowTarget.width != glowWidth || glowTarget.height != glowHeight )
    {
        bool complete = Create( glowTarget, glowWidth, glowHeight ) && Create( scratchTarget, glowWidth, glowHeight );
        rTextureRenderTarget::Restore();
        if ( !complete )
            return false;
    }
    return true;
}

static rCallbackBeforeScreenModeChange sr_bloomUnload( &Unload );

}

void sr_Bloom()
{
    if ( !sr_bloom || !sr_glOut || failed || sr_bloomIntensity <= 0 )
        return;
    int width = sr_screenWidth, height = sr_screenHeight;
    if ( width < 8 || height < 8 )
        return;
    if ( !GLEW_VERSION_2_0 || !GLEW_EXT_framebuffer_object )
    {
        Fail( "this OpenGL has no shaders or framebuffer objects" );
        return;
    }

    // close whatever the renderer wrapper still has open, forget stale errors
    RenderEnd();
    while ( glGetError() != GL_NO_ERROR )
        ;

    if ( !Prepare( width, height ) )
    {
        Fail( "the glow shaders or buffers could not be created" );
        return;
    }

    // 1. grab the finished scene (a multisampled window is resolved by the copy)
    glBindTexture( GL_TEXTURE_2D, sceneTexture );
    glCopyTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height );

    glPushAttrib( GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_VIEWPORT_BIT | GL_TEXTURE_BIT | GL_CURRENT_BIT | GL_POLYGON_BIT );
    glMatrixMode( GL_TEXTURE );
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    glDisable( GL_DEPTH_TEST );
    glDepthMask( GL_FALSE );
    glDisable( GL_ALPHA_TEST );
    glDisable( GL_BLEND );
    glDisable( GL_LIGHTING );
    glDisable( GL_CULL_FACE );
    glDisable( GL_FOG );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    glEnable( GL_TEXTURE_2D );
    glColor4f( 1, 1, 1, 1 );

    // 2. bright parts, glow buffer size
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, glowTarget.framebuffer );
    glViewport( 0, 0, glowTarget.width, glowTarget.height );
    glUseProgram( brightProgram );
    glUniform2f( brightTexel, 1.f / width, 1.f / height );
    glUniform1f( brightThreshold, sr_bloomThreshold );
    glUniform1i( brightTaps, Shrink( height ) / 2 );
    glBindTexture( GL_TEXTURE_2D, sceneTexture );
    Quad();

    // 3. blur, twice across and twice down
    glUseProgram( blurProgram );
    for ( int pass = 0; pass < 2; ++pass )
    {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, scratchTarget.framebuffer );
        glBindTexture( GL_TEXTURE_2D, glowTarget.texture );
        glUniform2f( blurStep, sr_bloomRadius / glowTarget.width, 0 );
        Quad();
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, glowTarget.framebuffer );
        glBindTexture( GL_TEXTURE_2D, scratchTarget.texture );
        glUniform2f( blurStep, 0, sr_bloomRadius / glowTarget.height );
        Quad();
    }
    glUseProgram( 0 );

    // 4. add the glow onto the scene
    rTextureRenderTarget::Restore();
    glViewport( 0, 0, width, height );
    glEnable( GL_BLEND );
    glBlendFunc( GL_ONE, GL_ONE );
    glBindTexture( GL_TEXTURE_2D, glowTarget.texture );
    glColor4f( sr_bloomIntensity, sr_bloomIntensity, sr_bloomIntensity, 1 );
    Quad();

    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
    glMatrixMode( GL_TEXTURE );
    glPopMatrix();
    glPopAttrib();
    glMatrixMode( GL_MODELVIEW );

    if ( glGetError() != GL_NO_ERROR )
        Fail( "an OpenGL error happened" );
}

#else // HAVE_GLEW

void sr_Bloom()
{
}

#endif // HAVE_GLEW

#else // DEDICATED

void sr_Bloom()
{
}

#endif // DEDICATED
