#include "OgreApp.h"
#include "OgreAppLogic.h"
#include <Ogre.h>

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
#define WIN32_LEAN_AND_MEAN
#include "windows.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if 0 //OGRE_PLATFORM == OGRE_PLATFORM_WIN32
INT WINAPI WinMain( HINSTANCE hInst, HINSTANCE, LPSTR strCmdLine, INT )
#else
int main(int argc, char **argv)
#endif
{
	try 
	{
		OgreApp app;
		OgreAppLogic appLogic;
		app.setAppLogic(&appLogic);
		//app.setCommandLine(Ogre::String(strCmdLine));
		app.run();
    }
	catch( Ogre::Exception& e ) 
	{
        MessageBox( NULL, e.getFullDescription().c_str(), "An exception has occured!", MB_OK | MB_ICONERROR | MB_TASKMODAL);
		return 1;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
