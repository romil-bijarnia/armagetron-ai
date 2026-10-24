// Exercise the production GLSL on macOS without opening a window.
// Regressions: overlapping ribbons retain their chroma, interiors remain
// coloured/translucent, and flow survives near, distant and mirrored views.
// Film-style pale highlights are allowed in the narrow edge cores.
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

std::string Read(char const *path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read shader source");
    std::ostringstream source;
    source << file.rdbuf();
    return source.str();
}

std::string Shader(std::string const &source, std::string const &tag)
{
    auto start=source.find(tag);
    if (start==source.npos) throw std::runtime_error("Shader declaration missing");
    start+=tag.size();
    auto end=source.find(")GLSL\"",start);
    if (end==source.npos) throw std::runtime_error("Shader terminator missing");
    return source.substr(start,end-start);
}

GLuint Compile(GLenum type, std::string const &source)
{
    GLuint shader=glCreateShader(type);
    auto text=source.c_str();
    glShaderSource(shader,1,&text,nullptr);
    glCompileShader(shader);
    GLint success=GL_FALSE;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&success);
    if (!success)
    {
        char log[8192]={};
        glGetShaderInfoLog(shader,sizeof(log),nullptr,log);
        throw std::runtime_error(log);
    }
    return shader;
}

void Scalar(GLuint program,char const *name,float value)
{
    glUniform1f(glGetUniformLocation(program,name),value);
}
void Pair(GLuint program,char const *name,float x,float y)
{
    glUniform2f(glGetUniformLocation(program,name),x,y);
}

int Check(char const *materialPath,char const *colourPath)
{
    auto source=Read(materialPath),shared=Read(colourPath);
    GLuint vertex=Compile(GL_VERTEX_SHADER,Shader(source,"char const * vertex=R\"GLSL("));
    GLuint fragment=Compile(GL_FRAGMENT_SHADER,Shader(shared,"return R\"GLSL(")+
        Shader(source,"char const * fragment=R\"GLSL("));
    GLuint program=glCreateProgram();
    glAttachShader(program,vertex);
    glAttachShader(program,fragment);
    glLinkProgram(program);
    GLint linked=GL_FALSE;
    glGetProgramiv(program,GL_LINK_STATUS,&linked);
    if (!linked) throw std::runtime_error("Material link failed");

    GLuint texture,framebuffer;
    glGenTextures(1,&texture);
    glBindTexture(GL_TEXTURE_2D,texture);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,512,512,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glGenFramebuffersEXT(1,&framebuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,framebuffer);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT0_EXT,GL_TEXTURE_2D,texture,0);
    if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT)
        throw std::runtime_error("Test framebuffer incomplete");

    // Bind a complete texture for the compatibility spill branch, independently
    // of the render target. No scene texture, window or screenshot is required.
    unsigned char white[4]={255,255,255,255};
    GLuint neutral;
    glGenTextures(1,&neutral);
    glBindTexture(GL_TEXTURE_2D,neutral);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,1,1,0,GL_RGBA,GL_UNSIGNED_BYTE,white);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glViewport(0,0,512,512);
    glUseProgram(program);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
    Pair(program,"bankSinCos",0,1);
    Pair(program,"rearEmitter",1000,1000);
    Pair(program,"bikeForward",1,0);
    Pair(program,"tyreShape",.7,.65);
    Scalar(program,"bikeScale",.825);
    Scalar(program,"headCoord",40);
    Pair(program,"referenceViewport",540,540);
    Scalar(program,"pixelScale",512.f/1080);

    std::array<std::array<float,3>,8> colours={{{1,0,0},{1,1.f/3,0},
        {1,11.f/15,0},{0,1,0},{1.f/15,13.f/15,1},{0,0,1},{2.f/3,0,1},{1,1,1}}};
    int cases=0,failures=0,animated=0;
    float worst=0,worstChroma=0;
    std::array<std::vector<unsigned char>,4> singleLayer;
    std::vector<unsigned char> first,pixels(512*512*4);
    for (auto tint:colours)
    for (int height:{8,24,72,224})
    for (int layers:{1,2,8,64})
    for (int mirror:{0,1})
    for (int tick=0;tick<2;++tick)
    {
        glClearColor(0,0,0,0);
        glClear(GL_COLOR_BUFFER_BIT);
        Scalar(program,"flowTime",tick*1.7);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0,4,0,1.22*512/height,-10,10);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        if (mirror) { glTranslatef(4,0,0);glScalef(-1,1,1); }
        glRotatef(-90,1,0,0);
        glColor4f(tint[0],tint[1],tint[2],1);
        for (int layer=0;layer<layers;++layer)
        {
            glBegin(GL_QUADS);
            glTexCoord4f(0,1,0,1.22);glVertex3f(0,0,0);
            glTexCoord4f(4,1,0,1.22);glVertex3f(4,0,0);
            glTexCoord4f(4,0,0,1.22);glVertex3f(4,0,1.22);
            glTexCoord4f(0,0,0,1.22);glVertex3f(0,0,1.22);
            glEnd();
        }
        glReadPixels(0,0,512,512,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        bool fail=false;
        for (int y=0;y<height;++y)
        for (int x=1;x<511;++x)
        {
            int i=(y*512+x)*4;
            float peak=std::max({pixels[i],pixels[i+1],pixels[i+2]})/255.f;
            float minimum=std::min({pixels[i],pixels[i+1],pixels[i+2]})/255.f;
            float excess=peak-pixels[i+3]/255.f;
            worst=std::max(worst,excess);
            // This bound prevents same-colour layers clipping independently
            // toward white, while allowing deliberately hotter edge colours.
            if(excess>2.f/255 || peak<.01) fail=true;
            for(int c=0;c<3;++c)
                for(int d=0;d<3;++d)
                    if(tint[c]>tint[d]+.05 && pixels[i+c]+2<pixels[i+d]) fail=true;
            bool interior=y>=height*.25 && y<height*.75;
            bool coloured=*std::min_element(tint.begin(),tint.end())<.3;
            if(interior && coloured && (peak-minimum)/std::max(peak,.001f)<.65) fail=true;
            if(layers==1 && interior && pixels[i+3]>200) fail=true;
            if(layers>1)
            {
                auto const &base=singleLayer[mirror*2+tick];
                float basePeak=std::max({base[i],base[i+1],base[i+2]});
                if(basePeak>32)
                    for(int c=0;c<3;++c)
                    {
                        float drift=std::abs(pixels[i+c]/(peak*255)-base[i+c]/basePeak);
                        worstChroma=std::max(worstChroma,drift);
                        if(drift>.04) fail=true;
                    }
            }
        }
        if(layers==1) singleLayer[mirror*2+tick]=pixels;
        if (!tick) first=pixels;
        else if (pixels!=first) ++animated;
        if (fail) ++failures;
        ++cases;
    }
    GLenum error=glGetError();
    std::cout << "{\"cases\":" << cases << ",\"colour_bound_failures\":" << failures
        << ",\"worst_premultiplied_excess\":" << worst
        << ",\"worst_overlap_chroma_drift\":" << worstChroma << ",\"animated_pairs\":" << animated
        << ",\"gl_error\":" << error << "}\n";
    return failures || error || animated!=cases/2 ? 1:0;
}

int main(int argc,char **argv)
{
    if (argc!=3) return 2;
    CGLPixelFormatAttribute attributes[]={kCGLPFAAccelerated,kCGLPFAColorSize,
        static_cast<CGLPixelFormatAttribute>(24),static_cast<CGLPixelFormatAttribute>(0)};
    CGLPixelFormatObj format=nullptr;
    CGLContextObj context;
    GLint count;
    if (CGLChoosePixelFormat(attributes,&format,&count)!=kCGLNoError || !format)
    {
        std::cerr << "Accelerated OpenGL context unavailable on this host\n";
        return 77;
    }
    auto error=CGLCreateContext(format,nullptr,&context);
    CGLDestroyPixelFormat(format);
    if (error!=kCGLNoError)
    {
        std::cerr << "Accelerated OpenGL context unavailable on this host\n";
        return 77;
    }
    if (CGLSetCurrentContext(context)!=kCGLNoError)
    {
        CGLDestroyContext(context);
        return 77;
    }
    int result=1;
    try { result=Check(argv[1],argv[2]); }
    catch (std::exception const &failure) { std::cerr << failure.what() << '\n'; }
    CGLSetCurrentContext(nullptr);
    CGLDestroyContext(context);
    return result;
}
