#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre.h>
#include <OgrePanelOverlayElement.h>

#include "StatsFrameListener.h"

using namespace Ogre;

OgreAppLogic::OgreAppLogic() 
: mApplication(0)
{
	// ogre
	mSceneMgr		     = 0;
	mViewport	     	 = 0;
	mStatsFrameListener  = 0;
	mOISListener.mParent = this;
	mTotalTime           = 0;	
	mTimeUntilNextToggle = 0;
	mKinectManager       = NULL;
	mKinect              = NULL;
	mOgreTexture.setNull();
	mKinectMotorPosition = 1.0;
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
	
	mKinectManager = new Ogre::Kinect::DeviceManager;
	unsigned int nbKinectConnected = mKinectManager->getKinectCount();
	if (nbKinectConnected > 0)
	{
		std::cout << nbKinectConnected << " Kinect connected" << std::endl;		
		mKinect = mKinectManager->getKinect(0);

		const std::string colorTextureName        = "KinectColorTexture";
		const std::string depthTextureName        = "KinectDepthTexture";
		const std::string coloredDepthTextureName = "KinectColoredDepthTexture";

		mKinect->createTexture(colorTextureName, depthTextureName);
		mKinect->createColoredDepthTexture(coloredDepthTextureName);
		mKinect->setMotorPosition(mKinectMotorPosition);

		createKinectOverlay(colorTextureName, depthTextureName, coloredDepthTextureName);
	}
	else
	{
		std::cout << "No Kinect connected !" << std::endl;
	}

	return true;
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	mKinect->update();
	bool result = processInputs(deltaTime);
	return result;
}

void OgreAppLogic::shutdown(void)
{
	mOgreTexture.setNull();
	
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
	mCamera->setPosition(Ogre::Vector3(0, 3, -2));
	mCamera->lookAt(0, 0, 0);
	mCamera->setNearClipDistance(0.1f);
	mCamera->setFarClipDistance(10000.0);
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
	const Degree ROT_SCALE = Degree(100.0f);
	const Real MOVE_SCALE = 5.0;
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

	if (keyboard->isKeyDown(OIS::KC_SUBTRACT) && mTimeUntilNextToggle <= 0)
	{		
		mKinectMotorPosition = max(0.0f, mKinectMotorPosition - 0.05f); 
		mKinect->setMotorPosition(mKinectMotorPosition);
		mTimeUntilNextToggle = 0.1f;
	}

	if (keyboard->isKeyDown(OIS::KC_ADD) && mTimeUntilNextToggle <= 0)
	{		
		mKinectMotorPosition = min(1.0f, mKinectMotorPosition + 0.05f);
		mKinect->setMotorPosition(mKinectMotorPosition);
		mTimeUntilNextToggle = 0.1f;
	}

	if (keyboard->isKeyDown(OIS::KC_SPACE) && mTimeUntilNextToggle <= 0)
	{
		mIsCudaEnabled = !mIsCudaEnabled;
		mTimeUntilNextToggle = 0.2f;
	}

	if (mTimeUntilNextToggle >= 0)
		mTimeUntilNextToggle -= deltaTime;


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

void OgreAppLogic::createKinectOverlay(const std::string& colorTextureName, const std::string& depthTextureName, const std::string& coloredDepthTextureName)
{
	//Create Color Overlay
	{
		//Create Overlay
		Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
		Ogre::Overlay* overlay = overlayManager.create("KinectColorOverlay");

		//Create Material
		const std::string materialName = "KinectColorMaterial";
		Ogre::MaterialPtr material = MaterialManager::getSingleton().create(materialName, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
		material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
		material->getTechnique(0)->getPass(0)->createTextureUnitState(colorTextureName);

		//Create Panel
		Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "KinectColorPanel"));
		panel->setMetricsMode(Ogre::GMM_PIXELS);
		panel->setMaterialName(materialName);
		panel->setDimensions((float)Ogre::Kinect::colorWidth, (float)Ogre::Kinect::colorHeight);
		panel->setPosition(640.0f, 0.0f);
		overlay->add2D(panel);		
		overlay->setZOrder(300);
		overlay->show(); 
	}

	//Create Depth Overlay
	{
		//Create Overlay
		Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
		Ogre::Overlay* overlay = overlayManager.create("KinectDepthOverlay");

		//Create Material
		const std::string materialName = "KinectDepthMaterial";
		Ogre::MaterialPtr material = MaterialManager::getSingleton().create(materialName, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
		material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
		material->getTechnique(0)->getPass(0)->setAlphaRejectSettings(CMPF_GREATER, 127);

		material->getTechnique(0)->getPass(0)->createTextureUnitState(depthTextureName);
		material->getTechnique(0)->getPass(0)->setVertexProgram("Ogre/Compositor/StdQuad_vp");
		material->getTechnique(0)->getPass(0)->setFragmentProgram("KinectDepth");

		//Create Panel
		Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "KinectDepthPanel"));
		panel->setMetricsMode(Ogre::GMM_PIXELS);
		panel->setMaterialName(materialName);
		panel->setDimensions((float)Ogre::Kinect::depthWidth, (float)Ogre::Kinect::depthHeight);
		panel->setPosition((float)640.0f, 0.0f);
		overlay->add2D(panel);		
		overlay->setZOrder(310);
		overlay->show();
	}

	//Create Colored Depth Overlay
	{
		//Create Overlay
		Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
		Ogre::Overlay* overlay = overlayManager.create("KinectColoredDepthOverlay");

		//Create Material
		const std::string materialName = "KinectColoredDepthMaterial";
		Ogre::MaterialPtr material = MaterialManager::getSingleton().create(materialName, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
		material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
		material->getTechnique(0)->getPass(0)->createTextureUnitState(coloredDepthTextureName);

		//Create Panel
		Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "KinectColoredDepthPanel"));
		panel->setMetricsMode(Ogre::GMM_PIXELS);
		panel->setMaterialName(materialName);
		panel->setDimensions((float)Ogre::Kinect::depthWidth, (float)Ogre::Kinect::depthHeight);
		panel->setPosition((float)0.0f, 0.0f);
		overlay->add2D(panel);		
		overlay->setZOrder(320);
		overlay->show();
	}
}