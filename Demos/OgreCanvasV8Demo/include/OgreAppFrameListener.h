#ifndef OGREOGREAPPFRAMELISTENER_H
#define OGREOGREAPPFRAMELISTENER_H

#include <Ogre/Ogre.h>

class OgreApp;

class OgreAppFrameListener : public Ogre::FrameListener, public Ogre::WindowEventListener
{
public:
	// Constructor takes a RenderWindow because it uses that to determine input context
	OgreAppFrameListener(OgreApp* app);
	virtual ~OgreAppFrameListener();

	//Adjust mouse clipping area
	virtual void windowResized(Ogre::RenderWindow* rw);
	virtual void windowMoved(Ogre::RenderWindow* rw);

	//Unattach OIS before window shutdown (very important under Linux)
	virtual void windowClosed(Ogre::RenderWindow* rw);

	// Override frameStarted event to process that (don't care about frameEnded)
	//bool frameStarted(const Ogre::FrameEvent& evt);
	bool frameRenderingQueued(const Ogre::FrameEvent& evt);

protected:
	OgreApp *mApplication;
	bool mWindowClosed;
};

#endif // OGREOGREAPPFRAMELISTENER_H