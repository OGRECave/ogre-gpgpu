#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre/Ogre.h>
#include "StatsFrameListener.h"

using namespace Ogre;

OgreAppLogic::OgreAppLogic() : mApplication(0)
{
	// ogre
	mSceneMgr		= 0;
	mViewport		= 0;
	mStatsFrameListener = 0;
	mTimeUntilNextToggle = 0;
	mOISListener.mParent = this;
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

	createCanvasOverlay();

	return true;
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	if (mTimeUntilNextToggle <= 0)
	{
		mCanvasJS->execute("update();");
		mTexture->uploadTexture();
		mTimeUntilNextToggle = 0.040f;
	}
	bool result = processInputs(deltaTime);
	return result;
}

void OgreAppLogic::shutdown(void)
{
	delete mCanvasJS;
	mTexture->deleteMaterial();
	delete mTexture;

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

}

//--------------------------------- update --------------------------------

bool OgreAppLogic::processInputs(Ogre::Real deltaTime)
{
	OIS::Keyboard *keyboard = mApplication->getKeyboard();

	if(keyboard->isKeyDown(OIS::KC_ESCAPE))
	{
		return false;
	}

	if (mTimeUntilNextToggle >= 0)
		mTimeUntilNextToggle -= deltaTime;

	return true;
}

bool OgreAppLogic::OISListener::mouseMoved(const OIS::MouseEvent &arg)
{
	return true;
}

bool OgreAppLogic::OISListener::mousePressed(const OIS::MouseEvent &arg, OIS::MouseButtonID id)
{
	return true;
}

bool OgreAppLogic::OISListener::mouseReleased(const OIS::MouseEvent &arg, OIS::MouseButtonID id)
{
	return true;
}

bool OgreAppLogic::OISListener::keyPressed(const OIS::KeyEvent &arg)
{
	if (arg.key == OIS::KC_LEFT || arg.key == OIS::KC_Q)
	{
		mParent->mCanvasJS->execute("key[0]=1; update();");
	}
	else if (arg.key == OIS::KC_RIGHT || arg.key == OIS::KC_D)
	{
		mParent->mCanvasJS->execute("key[1]=1; update();");
	}
	else if (arg.key == OIS::KC_UP || arg.key == OIS::KC_Z)
	{
		mParent->mCanvasJS->execute("key[2]=1; update();");
	}
	else if (arg.key == OIS::KC_DOWN || arg.key == OIS::KC_S)
	{
		mParent->mCanvasJS->execute("key[3]=1; update();");
	}
	return true;
}

bool OgreAppLogic::OISListener::keyReleased(const OIS::KeyEvent &arg)
{
	if (arg.key == OIS::KC_LEFT || arg.key == OIS::KC_Q)
	{
		mParent->mCanvasJS->execute("key[0]=0; update();");
	}
	else if (arg.key == OIS::KC_RIGHT || arg.key == OIS::KC_D)
	{
		mParent->mCanvasJS->execute("key[1]=0; update();");
	}
	else if (arg.key == OIS::KC_UP || arg.key == OIS::KC_Z)
	{
		mParent->mCanvasJS->execute("key[2]=0; update();");
	}
	else if (arg.key == OIS::KC_DOWN || arg.key == OIS::KC_S)
	{
		mParent->mCanvasJS->execute("key[3]=0; update();");
	}
	return true;
}

void OgreAppLogic::createCanvasOverlay(void)
{
	//create canvas	
	mLogger   = new Ogre::Canvas::Logger("canvasLog.txt");
	mTexture = new Ogre::Canvas::Texture("canvascape", 400, 300, true);
	mContext = mTexture->getContext();
	mCanvasJS = new Ogre::Canvas::V8Context(mContext, mLogger);
	mCanvasJS->execute(mCanvasJS->readScript("../../media/canvascape.js"));
	mTexture->uploadTexture();
	mTexture->createMaterial();	

	OverlayManager& overlayManager = OverlayManager::getSingleton();

	OverlayContainer* panel = static_cast<OverlayContainer*>(overlayManager.createOverlayElement("Panel", "canvascape"));
	panel->setMetricsMode(Ogre::GMM_PIXELS);
	panel->setPosition(0, 0);
	panel->setDimensions(400, 300);
	panel->setMaterialName(mTexture->getName());

	Overlay* overlay = overlayManager.create("canvascape");
	overlay->add2D(panel);
	overlay->show();
}