#include "OgreApp.h"
#include "OgreAppLogic.h"
#include <Ogre.h>

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
#define WIN32_LEAN_AND_MEAN
#include "windows.h"
#endif

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
INT WINAPI WinMain( HINSTANCE hInst, HINSTANCE, LPSTR strCmdLine, INT )
#else
int main(int argc, char **argv)
#endif
{
    OgreApp app;
    OgreAppLogic appLogic;
    app.setAppLogic(&appLogic);
    //app.setCommandLine(Ogre::String(strCmdLine));
    app.run();

    return 0;
}
