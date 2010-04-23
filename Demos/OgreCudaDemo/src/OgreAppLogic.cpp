#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre.h>
#include <OgrePanelOverlayElement.h>

#include "StatsFrameListener.h"

using namespace Ogre;

extern "C" void cudaTextureUpdate(void* deviceTexture, int width, int height, float t);
extern "C" void cudaMeshUpdate(void* deviceMesh, int width, int height, float t);

OgreAppLogic::OgreAppLogic() 
: mApplication(0)
{
	// ogre
	mSceneMgr		= 0;
	mViewport		= 0;
	mStatsFrameListener = 0;
	mOISListener.mParent = this;
	mTotalTime = 0;
	mOgreTexture.setNull();
	mIsCudaEnabled = true;
	mTimeUntilNextToggle = 0;
	mCudaVertexBuffer = NULL;
	mCudaVertexBufferRenderable = NULL;
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
	
	int deviceCount = Ogre::Cuda::Root::getDeviceCount();
	for (int i=0; i<deviceCount; ++i)
		std:: cout << Ogre::Cuda::Root::getDeviceProperties(i) << std::endl;

	mCudaRoot = Ogre::Cuda::Root::createRoot(mApplication->getRenderWindow(), mApplication->getOgreRoot()->getRenderSystem());
	mCudaRoot->init();

	int width  = 128;
	int height = 128;
	createCudaMaterial(width, height);
	createCudaPlane(width, height);

	//Create Ogre::Cuda::Texture	
	mCudaTexture = mCudaRoot->getTextureManager()->createTexture(mOgreTexture);
	mCudaTexture->registerForCudaUse();

	//Test to ignore D3D10 (because VertexBuffer are not working)
	if (mCudaRoot->getVertexBufferManager())
	{
		mCudaVertexBufferRenderable = new CudaVertexBufferRenderable(128, 128);	
		Ogre::SceneNode* cudaNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("CudaMesh");
		cudaNode->attachObject(mCudaVertexBufferRenderable);

		//Create Ogre::Cuda::VertexBuffer
		mCudaVertexBuffer = mCudaRoot->getVertexBufferManager()->createVertexBuffer(mCudaVertexBufferRenderable->getHardwareVertexBuffer());
		mCudaVertexBuffer->registerForCudaUse();
	}

	return true;
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	if (mIsCudaEnabled)
	{
		mTotalTime += deltaTime;

		mCudaTexture->map();
		Ogre::Cuda::TextureDeviceHandle textureHandle = mCudaTexture->getDeviceHandle(0, 0);
		cudaTextureUpdate(textureHandle.getPointer(), textureHandle.width, textureHandle.height, mTotalTime);		
		mCudaTexture->update(textureHandle);
		mCudaTexture->unmap();

		if (mCudaVertexBuffer)
		{
			mCudaVertexBuffer->map();
			void* deviceMesh = mCudaVertexBuffer->getPointer();
			cudaMeshUpdate(deviceMesh, 128, 128, mTotalTime);
			mCudaVertexBuffer->unmap();
		}
	}

	bool result = processInputs(deltaTime);
	return result;
}

void OgreAppLogic::shutdown(void)
{

	mCudaTexture->unregister();
	mCudaRoot->getTextureManager()->destroyTexture(mCudaTexture);
	mCudaVertexBuffer->unregister();
	mCudaRoot->getVertexBufferManager()->destroyVertexBuffer(mCudaVertexBuffer);
	mCudaRoot->shutdown();
	Ogre::Cuda::Root::destroyRoot(mCudaRoot);
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
	mSceneMgr->setSkyBox(true, "Examples/Grid");
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

void OgreAppLogic::createCudaPlane(int width, int height)
{
	//Create plane
	Ogre::MeshManager::getSingleton().createPlane("CudaPlane", 
		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		Ogre::Plane(Ogre::Vector3::UNIT_Y, Ogre::Vector3::ZERO), 1, 1, 
		4, 4, true, 1, 1, 1, Vector3::UNIT_Z);
	
	//Create entity
	Ogre::Entity* planeEntity = mSceneMgr->createEntity("CudaGridPlane", "CudaPlane");
	planeEntity->setMaterialName("CudaMaterial");
	planeEntity->setRenderQueueGroup(RENDER_QUEUE_WORLD_GEOMETRY_1);

	mSceneMgr->getRootSceneNode()->attachObject(planeEntity);
}

void OgreAppLogic::createCudaMaterial(int width, int height)
{
	//Create Ogre::Texture
	mOgreTexture  = Ogre::TextureManager::getSingleton().createManual("CudaTexture",  
		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		Ogre::TEX_TYPE_2D, 
		width, 
		height, 
		0, 
		Ogre::PF_A8R8G8B8, 
		Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

	//Create Ogre::Material
	Ogre::MaterialPtr material = MaterialManager::getSingleton().create("CudaMaterial", 
		Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);

	Ogre::Technique *technique = material->createTechnique();
	technique->createPass();
	material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
	material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
	material->getTechnique(0)->getPass(0)->createTextureUnitState("CudaTexture");
}