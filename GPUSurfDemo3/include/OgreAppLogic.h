/*
	Copyright (c) 2010 ASTRE Henri (http://www.visual-experiments.com)

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

#ifndef OGREAPPLOGIC_H
#define OGREAPPLOGIC_H

#include <Ogre/OgrePrerequisites.h>
#include <Ogre/OgreStringVector.h>
#include <OIS/OIS.h>

#include "GPUSurf.h"
#include <CanvasTexture.h>
#include "VideoDeviceManager.h"

class OgreApp;
class StatsFrameListener;

class OgreAppLogic
{
public:
	OgreAppLogic();
	~OgreAppLogic();

	void setParentApp(OgreApp *app) { mApplication = app; }

	/// Called before Ogre and everything in the framework is initialized
	/// Configure the framework here
	bool preInit(const Ogre::StringVector &commandArgs);
	/// Called when Ogre and the framework is initialized
	/// Init the logic here
	bool init(void);

	/// Called before everything in the framework is updated
	bool preUpdate(Ogre::Real deltaTime);
	/// Called when the framework is updated
	/// update the logic here
	bool update(Ogre::Real deltaTime);

	/// Called before Ogre and the framework are shut down
	/// shutdown the logic here
	void shutdown(void);
	/// Called when Ogre and the framework are shut down
	void postShutdown(void);

	void createSceneManager(void);
	void createViewport(void);
	void createCamera(void);
	void createScene(void);

	bool processInputs(Ogre::Real deltaTime);

protected:
	// OGRE
	OgreApp *mApplication;
	Ogre::SceneManager *mSceneMgr;
	Ogre::Viewport *mViewport;
	Ogre::Camera *mCamera;
	VideoDevice* mVideoDevice;
	VideoDeviceManager mVideoDeviceManager;
	int mVideoWidth;
	int mVideoHeight;
	int mNbOctave;
	Ogre::Canvas::Texture* mCanvasTexture;

	StatsFrameListener *mStatsFrameListener;

	void createCanvasOverlay(int width, int height, const std::string& materialName);
	void displayDetectedFeaturesPerOctave(Feature* features, unsigned int nbFeature);

	// OIS
	class OISListener : public OIS::MouseListener, public OIS::KeyListener
	{
	public:
		virtual bool mouseMoved( const OIS::MouseEvent &arg );
		virtual bool mousePressed( const OIS::MouseEvent &arg, OIS::MouseButtonID id );
		virtual bool mouseReleased( const OIS::MouseEvent &arg, OIS::MouseButtonID id );
		virtual bool keyPressed( const OIS::KeyEvent &arg );
		virtual bool keyReleased( const OIS::KeyEvent &arg );
		OgreAppLogic *mParent;
	};
	friend class OISListener;
	OISListener mOISListener;

	Ogre::Real mTimeUntilNextToggle;
	void createWebcamOverlay();
	GPUSurf*           mGPUSurf;	
};

#endif // OGREAPPLOGIC_H