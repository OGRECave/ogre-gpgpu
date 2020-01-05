#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre/Ogre.h>
#include "StatsFrameListener.h"
#include "CanvasV8Context.h"

using namespace Ogre;

OgreAppLogic::OgreAppLogic() 
: mApplication(0)
{
	// ogre
	mSceneMgr		= 0;
	mViewport		= 0;
	mDemoIndex      = 0;
	mStatsFrameListener = 0;
	mTimeUntilNextToggle = 0;
	mPanel = 0;
	mTexture = 0;
	mOISListener.mParent = this;
	mLogger = new Ogre::Canvas::Logger("Canvas.log");
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
	createCanvasOverlay();

	mStatsFrameListener = new StatsFrameListener(mApplication->getRenderWindow());
	mApplication->getOgreRoot()->addFrameListener(mStatsFrameListener);
	mStatsFrameListener->showDebugOverlay(true);

	mApplication->getKeyboard()->setEventCallback(&mOISListener);
	mApplication->getMouse()->setEventCallback(&mOISListener);

	if (mDemoViewer.getNbDemo() > 0)
		setDemo(mDemoViewer.getDemo(0));

	return true;
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	bool result = processInputs(deltaTime);
	return result;
}

void OgreAppLogic::shutdown(void)
{
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
		nextDemo();
		mTimeUntilNextToggle = 0.2f;
	}
	else if (keyboard->isKeyDown(OIS::KC_SUBTRACT) && mTimeUntilNextToggle <= 0)
	{
		previousDemo();
		mTimeUntilNextToggle = 0.2f;
	}

	if (mTimeUntilNextToggle >= 0)
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

void OgreAppLogic::nextDemo()
{
	mDemoIndex++;
	if (mDemoIndex >= mDemoViewer.getNbDemo())
		mDemoIndex = 0;

	setDemo(mDemoViewer.getDemo(mDemoIndex));
}

void OgreAppLogic::previousDemo()
{
	mDemoIndex--;
	if (mDemoIndex < 0)
		mDemoIndex = mDemoViewer.getNbDemo()-1;

	setDemo(mDemoViewer.getDemo(mDemoIndex));
}

void OgreAppLogic::setDemo(const Demo& _demo)
{
	mPanel->setMaterialName("Examples/Grid");
	createCanvasMaterial("Canvas", _demo.width, _demo.height);
	mPanel->setDimensions((Ogre::Real)_demo.width, (Ogre::Real)_demo.height);
	mPanel->setMaterialName("Canvas");
	emptyCanvas();
	fillCanvas(_demo.script);
}

void OgreAppLogic::createCanvasMaterial(const std::string& _name, int _width, int _height)
{
	if (mTexture != NULL)
		deleteCanvasMaterial();

	mTexture = new Ogre::Canvas::Texture(_name, _width, _height, true);
	mContext = mTexture->getContext();

	mTexture->uploadTexture();
	mTexture->createMaterial();
}

void OgreAppLogic::deleteCanvasMaterial()
{
	mTexture->deleteMaterial();
	delete mTexture;
	mTexture = NULL;
	mContext = NULL;
}

void OgreAppLogic::fillCanvas(const std::string& _scriptFilename)
{
	Ogre::Canvas::V8Context canvasJS(mContext, mLogger);
	canvasJS.execute(canvasJS.readScript(_scriptFilename));

	mTexture->uploadTexture();
}

void OgreAppLogic::emptyCanvas()
{
	Ogre::Canvas::Context* ctx = mContext;
	ctx->save();
	ctx->fillStyle(Ogre::ColourValue::White);
	ctx->fillRect(0, 0, 150, 200);
	ctx->fillStyle(Ogre::ColourValue::Black);
	ctx->restore();
	mTexture->uploadTexture();
}

void OgreAppLogic::createCanvasOverlay()
{
	//create Texture
	mTexture = new Ogre::Canvas::Texture("Canvas", 640, 480, true, 0);

	//create Material
	mTexture->createMaterial();

	//create Overlay
	Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
	Ogre::Overlay* overlay = overlayManager.create("Canvas/Overlay");

	mPanel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "Canvas/Panel"));
	mPanel->setMetricsMode(Ogre::GMM_PIXELS);
	mPanel->setMaterialName("Canvas");
	mPanel->setDimensions(640, 480);
	mPanel->setPosition(0, 0);
	overlay->add2D(mPanel);

	overlay->show();
}