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

#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre/Ogre.h>
#include <Ogre/OgrePanelOverlayElement.h>
#include <Ogre/OgreExternalTextureSourceManager.h>

#include "StatsFrameListener.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace Ogre;

OgreAppLogic::OgreAppLogic() 
: mApplication(0)
{
	// ogre
	mSceneMgr		= 0;
	mViewport		= 0;
	mStatsFrameListener = 0;
	mOISListener.mParent = this;
	mTimeUntilNextToggle = 0;
	mVideoWidth  = 640;
	mVideoHeight = 480;
	mNbOctave    = 4;
}

OgreAppLogic::~OgreAppLogic()
{}

// preAppInit
bool OgreAppLogic::preInit(const Ogre::StringVector &commandArgs)
{
	return true;
}

// postAppInit
bool OgreAppLogic::init(void)
{
	createSceneManager();
	createViewport();
	createCamera();
	createScene();

	mStatsFrameListener = new StatsFrameListener(mApplication->getRenderWindow());
	mApplication->getOgreRoot()->addFrameListener(mStatsFrameListener);
	mStatsFrameListener->showDebugOverlay(true);

	mApplication->getKeyboard()->setEventCallback(&mOISListener);
	mApplication->getMouse()->setEventCallback(&mOISListener);

	createWebcamOverlay();

	mGPUSurf = new GPUSurf(mNbOctave);
	mGPUSurf->init(mVideoWidth, mVideoHeight);
	mGPUSurf->createOverlays();
	int width = 0;
	for (int j=0; j<mNbOctave; ++j)
		width += mVideoWidth / (1<<j);
	mCanvasTexture = new Ogre::Canvas::Texture("FeatureFound", width, mVideoHeight);
	mCanvasTexture->createMaterial();
	createCanvasOverlay(width, mVideoHeight, "FeatureFound");

 	return true;
}

void OgreAppLogic::createWebcamOverlay()
{
	if (mVideoDeviceManager.size() > 0)
	{
		int width  = mVideoWidth;
		int height = mVideoHeight;
		mVideoDevice = mVideoDeviceManager[0];
		mVideoDevice->init(width, height, 30);
		mVideoDevice->createTexture("WebcamVideoTexture");

		//Create Webcam Material
		MaterialPtr material = MaterialManager::getSingleton().create("WebcamVideoMaterial", ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		Ogre::Technique *technique = material->createTechnique();
		technique->createPass();
		material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
		material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
		material->getTechnique(0)->getPass(0)->createTextureUnitState("WebcamVideoTexture");
		/*
		//Create Webcam Overlay
		Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
		Ogre::Overlay* overlay = overlayManager.create("Webcam/Overlay");
		Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "Webcam/Panel"));
		panel->setMetricsMode(Ogre::GMM_PIXELS);
		panel->setMaterialName("WebcamVideoMaterial");
		panel->setDimensions((float)width, (float)height);
		panel->setPosition(0.0f, 0.0f);
		overlay->add2D(panel);
		overlay->show();
		*/
	}
	else
	{
		Ogre::Exception(Ogre::Exception::ERR_INVALID_STATE, "No webcam found", "AppLogic");
	}
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	if (mVideoDevice->update())
	{
		Ogre::PixelBox box(mVideoDevice->getWidth(), mVideoDevice->getHeight(), 1, Ogre::PF_B8G8R8, (void*) mVideoDevice->getBufferData());
		mGPUSurf->update(box);
		displayDetectedFeaturesPerOctave(mGPUSurf->getFeatures(), mGPUSurf->getNbFeatureFound());
	}
	bool result = processInputs(deltaTime);
	return result;
}

void OgreAppLogic::shutdown(void)
{
	delete mGPUSurf; 

	mVideoDevice->shutdown();
	mVideoDevice = NULL;

	mApplication->getOgreRoot()->removeFrameListener(mStatsFrameListener);
	delete mStatsFrameListener;
	mStatsFrameListener = 0;
	
	if(mSceneMgr)
		mApplication->getOgreRoot()->destroySceneManager(mSceneMgr);
	mSceneMgr = 0;
}

void OgreAppLogic::postShutdown(void)
{

}

//--------------------------------- Init --------------------------------

void OgreAppLogic::createSceneManager(void)
{
	mSceneMgr = mApplication->getOgreRoot()->createSceneManager(ST_GENERIC, "SceneManager");
}

void OgreAppLogic::createViewport(void)
{
	mViewport = mApplication->getRenderWindow()->addViewport(0);
}

void OgreAppLogic::createCamera(void)
{
	mCamera = mSceneMgr->createCamera("Camera");
	mCamera->setAutoAspectRatio(true);
	mViewport->setCamera(mCamera);
}

void OgreAppLogic::createScene(void)
{
	mSceneMgr->setAmbientLight(ColourValue(0.5,0.5,0.5));
	mSceneMgr->setSkyBox(true, "Examples/GridSkyBox");
}

//--------------------------------- update --------------------------------

bool OgreAppLogic::processInputs(Ogre::Real deltaTime)
{
	const Degree ROT_SCALE = Degree(60.0f);
	const Real MOVE_SCALE = 5000.0;
	Vector3 translateVector(Vector3::ZERO);
	Degree rotX(0);
	Degree rotY(0);
	OIS::Keyboard *keyboard = mApplication->getKeyboard();
	OIS::Mouse *mouse = mApplication->getMouse();

	if(keyboard->isKeyDown(OIS::KC_ESCAPE))
	{
		return false;
	}

	//////// moves  //////

	// keyboard moves
	if(keyboard->isKeyDown(OIS::KC_A))
		translateVector.x = -MOVE_SCALE;	// Move camera left

	if(keyboard->isKeyDown(OIS::KC_D))
		translateVector.x = +MOVE_SCALE;	// Move camera RIGHT

	if(keyboard->isKeyDown(OIS::KC_UP) || keyboard->isKeyDown(OIS::KC_W) )
		translateVector.z = -MOVE_SCALE;	// Move camera forward

	if(keyboard->isKeyDown(OIS::KC_DOWN) || keyboard->isKeyDown(OIS::KC_S) )
		translateVector.z = +MOVE_SCALE;	// Move camera backward

	if(keyboard->isKeyDown(OIS::KC_PGUP))
		translateVector.y = +MOVE_SCALE;	// Move camera up

	if(keyboard->isKeyDown(OIS::KC_PGDOWN))
		translateVector.y = -MOVE_SCALE;	// Move camera down

	if(keyboard->isKeyDown(OIS::KC_RIGHT))
		rotX -= ROT_SCALE;					// Turn camera right

	if(keyboard->isKeyDown(OIS::KC_LEFT))
		rotX += ROT_SCALE;					// Turn camea left

	// mouse moves
	const OIS::MouseState &ms = mouse->getMouseState();
	if (ms.buttonDown(OIS::MB_Right))
	{
		translateVector.x += ms.X.rel * 0.3f * MOVE_SCALE;	// Move camera horizontaly
		translateVector.y -= ms.Y.rel * 0.3f * MOVE_SCALE;	// Move camera verticaly
	}
	else
	{
		rotX += Degree(-ms.X.rel * 0.3f * ROT_SCALE);		// Rotate camera horizontaly
		rotY += Degree(-ms.Y.rel * 0.3f * ROT_SCALE);		// Rotate camera verticaly
	}	

	rotX *= deltaTime;
	rotY *= deltaTime;
	translateVector *= deltaTime;

	mCamera->moveRelative(translateVector);
	mCamera->yaw(rotX);
	mCamera->pitch(rotY);

	if (keyboard->isKeyDown(OIS::KC_ADD) && mTimeUntilNextToggle <= 0)
	{
		mTimeUntilNextToggle = 0.05f;
		mGPUSurf->up();
	}
	else if (keyboard->isKeyDown(OIS::KC_SUBTRACT) && mTimeUntilNextToggle <= 0)
	{
		mTimeUntilNextToggle = 0.05f;
		mGPUSurf->down();
	}

	if (mTimeUntilNextToggle > 0)
		mTimeUntilNextToggle -= deltaTime;

	return true;
}

bool OgreAppLogic::OISListener::mouseMoved( const OIS::MouseEvent &arg )
{
	return true;
}

bool OgreAppLogic::OISListener::mousePressed( const OIS::MouseEvent &arg, OIS::MouseButtonID id )
{
	return true;
}

bool OgreAppLogic::OISListener::mouseReleased( const OIS::MouseEvent &arg, OIS::MouseButtonID id )
{
	return true;
}

bool OgreAppLogic::OISListener::keyPressed( const OIS::KeyEvent &arg )
{
	return true;
}

bool OgreAppLogic::OISListener::keyReleased( const OIS::KeyEvent &arg )
{
	return true;
}

void OgreAppLogic::displayDetectedFeaturesPerOctave(Feature* features, unsigned int nbFeature)
{
	int width = 0;
	for (int j=0; j<mNbOctave; ++j)
		width += mVideoWidth / (1<<j);

	Ogre::Canvas::Context* ctx = mCanvasTexture->getContext();

	ctx->clearRect(0, 0, (float)width, 480.0f);
	for (unsigned int i=0; i<nbFeature; ++i)
	{
		Feature* feature = &features[i];

		if (feature->octave == 0)
			ctx->fillStyle(Ogre::ColourValue::Red);
		else if (feature->octave == 1)
			ctx->fillStyle(Ogre::ColourValue::Green);
		else if (feature->octave == 2)
			ctx->fillStyle(Ogre::ColourValue::Blue);
		else if (feature->octave == 3)
			ctx->fillStyle(Ogre::ColourValue::Black);
		else //if (feature->octave == 4)
			ctx->fillStyle(Ogre::ColourValue::White);
		{
			Ogre::Real scaleFactor = (Ogre::Real) (1<<(int)feature->octave);
			Ogre::Real offset = 0; 
			//for (unsigned int j=0; j<feature->octave; ++j)
			//	offset += mVideoWidth / (1<<j);
			ctx->fillRect(feature->x*scaleFactor-1 + offset, feature->y*scaleFactor,   3.0f, 1.0f);
			ctx->fillRect(feature->x*scaleFactor   + offset, feature->y*scaleFactor-1, 1.0f, 3.0f);
		}
	} 

	mCanvasTexture->uploadTexture();
}

void OgreAppLogic::createCanvasOverlay(int width, int height, const std::string& materialName)
{
	Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
	Ogre::Overlay* overlay = overlayManager.create("FeatureFound/Overlay");
	Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "FeatureFound/Panel"));
	panel->setMetricsMode(Ogre::GMM_PIXELS);
	panel->setMaterialName(materialName);
	panel->setDimensions((float)width, (float)height);
	panel->setPosition(0.0f, 0.0f);
	overlay->add2D(panel);
	overlay->show();
	overlay->setZOrder(650);
}