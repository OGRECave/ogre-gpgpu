#include "OgreAppFrameListener.h"
#include "OgreApp.h"
#include <Ogre.h>

using namespace Ogre;

// Constructor takes a RenderWindow because it uses that to determine input context
OgreAppFrameListener::OgreAppFrameListener(OgreApp* app) : mApplication(app)
{
	Ogre::RenderWindow *window = mApplication->getRenderWindow();

	if(window)
	{
		windowResized(window);
		WindowEventUtilities::addWindowEventListener(window, this);
	}

	mWindowClosed = false;
}

OgreAppFrameListener::~OgreAppFrameListener()
{
	Ogre::RenderWindow *window = mApplication->getRenderWindow();

	WindowEventUtilities::removeWindowEventListener(window, this);
	if(!window->isClosed())
		windowClosed(window);
}


//Adjust mouse clipping area
void OgreAppFrameListener::windowResized(RenderWindow* rw)
{
	windowMoved(rw);
}

void OgreAppFrameListener::windowMoved(RenderWindow* rw)
{
	unsigned int width, height, depth;
	int left, top;
	rw->getMetrics(width, height, depth, left, top);
	mApplication->notifyWindowMetrics(rw, left, top, width, height);
}

//Unattach OIS before window shutdown (very important under Linux)
void OgreAppFrameListener::windowClosed(RenderWindow* rw)
{
	mApplication->notifyWindowClosed(rw);
	mWindowClosed = true;
}

// Override frameStarted event to process that (don't care about frameEnded)
//bool OgreAppFrameListener::frameStarted(const FrameEvent& evt)
bool OgreAppFrameListener::frameRenderingQueued(const FrameEvent& evt)
{
	if(mWindowClosed)
		return false;

	if(!mApplication->update(evt.timeSinceLastFrame))
		return false;

	return true;
}
