#include "GPUSurfApplication.h"
#include "OgreExternalTextureSourceManager.h"
#include <sstream>

using namespace Ogre;
using namespace std;

GPUSurfFrameListener::GPUSurfFrameListener(Ogre::RenderWindow* window, Ogre::Camera* maincam, GPUSurfApplication* _app) 
: ExampleFrameListener(window, maincam), mApp(_app)
{
	showDebugOverlay(false);
}

bool GPUSurfFrameListener::processUnbufferedKeyInput(const  Ogre::FrameEvent& evt)
{
	if( mKeyboard->isKeyDown(OIS::KC_ADD) && mTimeUntilNextToggle <= 0 )
	{
		mTimeUntilNextToggle = 0.05;
		mApp->nextStage();
	}
	else if( mKeyboard->isKeyDown(OIS::KC_SUBTRACT) && mTimeUntilNextToggle <= 0 )
	{
		mTimeUntilNextToggle = 0.05;
		mApp->prevStage();
	}
	return ExampleFrameListener::processUnbufferedKeyInput(evt);
}

GPUSurfApplication::GPUSurfApplication()
{
	mStageNumber = 0;
	mGPUSurf     = new GPUSurf(5);
}

GPUSurfApplication::~GPUSurfApplication()
{
	delete mGPUSurf;
}

void GPUSurfApplication::createCamera(void)
{	
	mCamera = mSceneMgr->createCamera("camera");
	mCamera->setNearClipDistance(5);
	mCamera->setFarClipDistance(50000);
	mCamera->setPosition(0, 0, 0);
	mCamera->lookAt(0, 0, 1);
	mCamera->setFOVy(Degree(40));

	mCameraNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("cameraNode");
	mCameraNode->setPosition(0, 1700, 0);
	mCameraNode->lookAt(Vector3(0, 1700, -1), Node::TS_WORLD);
	mCameraNode->attachObject(mCamera);
	mCameraNode->setFixedYawAxis(true, Vector3::UNIT_Y);
}

void GPUSurfApplication::createScene(void) 
{
	//configure camera
	Viewport* vp = mWindow->getViewport(0);
	mCamera->setAspectRatio((float) vp->getActualWidth() / (float) vp->getActualHeight());	
	vp->setCamera(mCamera);

	mSceneMgr->setAmbientLight(Ogre::ColourValue(0.2, 0.2, 0.2));
	mSceneMgr->setSkyBox(true, "Examples/Grid", 50);
	mSceneMgr->setAmbientLight(Ogre::ColourValue(0.6,0.6,0.6));
	mSceneMgr->setShadowTechnique(Ogre::SHADOWTYPE_STENCIL_ADDITIVE);

    Ogre::Light* light =  mSceneMgr->createLight("SunLight");
	light->setPosition(Ogre::Vector3(150, 100, -150) );

	mSceneMgr->getRootSceneNode()->createChild("mesh")->setScale(20, 20, 20);
	
	createWebcamMaterial();
	createWebcamPlane(300.0f); //45000.0f

	Ogre::Vector2 dim = mVideo->getDimensions();
	mGPUSurf->init(dim.x, dim.y);
	mGPUSurf->createOverlays();
}

void GPUSurfApplication::destroyScene(void)
{
}

void GPUSurfApplication::createFrameListener(void) 
{
    mFrameListener = new GPUSurfFrameListener(mWindow, mCamera, this);
    mRoot->addFrameListener(mFrameListener);
}

void GPUSurfApplication::createWebcamMaterial()
{
	Ogre::DShowTextureSource *videoTextureSource = (DShowTextureSource *)Ogre::ExternalTextureSourceManager::getSingleton().getExternalTextureSource("dshow");
	if(videoTextureSource == NULL)
	{
		OGRE_EXCEPT(Exception::ERR_FILE_NOT_FOUND, "Could not find Plugin_DShow.dll", "GPUSurfApplication::createWebcamMaterial()");
	}

	try
	{
		//create texture
		videoTextureSource->initialise();
		//videoTextureSource->setInputFileName(videoFileName);		
		videoTextureSource->setInputCaptureDevice(0);
		videoTextureSource->setFPS(90);
		videoTextureSource->setVideoSize(640, 480);
		mVideo = videoTextureSource->_createVideo("WebcamVideoTexture");
		mVideo->createTexture("WebcamVideoTexture");		
		mVideo->load();
		mVideo->start();		
		mVideo->addListener(mGPUSurf);	

		//create material
		Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().create("WebcamVideoMaterial", "General");
		Pass* pass = material->getTechnique(0)->getPass(0);
		pass->setDepthCheckEnabled(false);
		pass->setDepthWriteEnabled(false);
		pass->setLightingEnabled(false);		
		pass->setSceneBlending(SBT_TRANSPARENT_ALPHA);		
		TextureUnitState* texUnit = pass->createTextureUnitState("WebcamVideoTexture");
		texUnit->setTextureFiltering(FO_POINT, FO_POINT, FO_POINT);	
	}
	catch (const InternalErrorException &)
	{
		OGRE_EXCEPT(Exception::ERR_INTERNAL_ERROR, "Could not init webcam", "GPUSurfApplication::createWebcamMaterial()");
	}
}

void GPUSurfApplication::createWebcamPlane(Real _distanceFromCamera)
{
	// Create a prefab plane dedicated to display video
    Vector2 videoDim = mVideo->getDimensions();
	float videoAspectRatio = videoDim.x / videoDim.y;

	float planeHeight = 2 * _distanceFromCamera * Ogre::Math::Tan(Degree(26.53)*0.5);
	float planeWidth = planeHeight * videoAspectRatio;

    Plane p(Vector3::UNIT_Z, 0.0);
	MeshManager::getSingleton().createPlane("VerticalPlane", ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, p , planeWidth, planeHeight, 1, 1, true, 1, 1, 1, Vector3::UNIT_Y);
	Entity* planeEntity = mSceneMgr->createEntity("VideoPlane", "VerticalPlane"); 
	planeEntity->setMaterialName("WebcamVideoMaterial");
	planeEntity->setRenderQueueGroup(RENDER_QUEUE_WORLD_GEOMETRY_1);

	// Create a node for the plane, inserts it in the scene
	Ogre::SceneNode* node = mCameraNode->createChildSceneNode("planeNode");
	node->attachObject(planeEntity);

    // Update position    
	Vector3 planePos = mCamera->getPosition() + mCamera->getDirection() * _distanceFromCamera;
	node->setPosition(planePos);

	// Update orientation
	node->setOrientation(mCamera->getOrientation());
}

void GPUSurfApplication::nextStage()
{
	mGPUSurf->up();
}

void GPUSurfApplication::prevStage()
{
	mGPUSurf->down();
}