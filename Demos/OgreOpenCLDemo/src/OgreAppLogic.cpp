#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre.h>
#include <OgrePanelOverlayElement.h>

#include "StatsFrameListener.h"
#include <OgreGLHardwareVertexBuffer.h>
#include <OgreD3D9HardwareVertexBuffer.h>
#include <OgreRenderSystemCapabilitiesManager.h>

using namespace Ogre;

#include <oclUtils.h>

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;
size_t szGlobalWorkSize[] = {mesh_width, mesh_height};
cl_kernel ckKernel;
cl_program cpProgram;

OgreAppLogic::OgreAppLogic() 
: mApplication(0)
{
	// ogre
	mSceneMgr		= 0;
	mViewport		= 0;
	mStatsFrameListener = 0;
	mOISListener.mParent = this;
	mTotalTime = 0;
	mIsOpenCLEnabled = true;
	mTimeUntilNextToggle = 0;
	mCustomVertexBufferRenderable = NULL;
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
	cl_int error;

	createSceneManager();
	createViewport();
	createCamera();
	createScene();

	mStatsFrameListener = new StatsFrameListener(mApplication->getRenderWindow());
	mApplication->getOgreRoot()->addFrameListener(mStatsFrameListener);
	mStatsFrameListener->showDebugOverlay(true);

	mApplication->getKeyboard()->setEventCallback(&mOISListener);
	mApplication->getMouse()->setEventCallback(&mOISListener);

	mCustomVertexBufferRenderable = new CustomVertexBufferRenderable(256, 256);	
	Ogre::SceneNode* clNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("OpenCLMesh");
	clNode->attachObject(mCustomVertexBufferRenderable);

	mCLRoot = Ogre::OpenCL::Root::createRoot(mApplication->getRenderWindow(), mApplication->getOgreRoot()->getRenderSystem());
	mCLRoot->init();

	// load program source
	size_t program_length;
	char *programSource = oclLoadProgSource("C:\\AnimatedTexture.cl", "", &program_length);
	
	// create the program
	cpProgram = clCreateProgramWithSource(*mCLRoot->getContext(), 1, (const char**) &programSource, &program_length, &error);

	// build the program
	error = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);

	// create the kernel
	ckKernel = clCreateKernel(cpProgram, "sine_wave", &error);

	// create vbo
	Ogre::HardwareVertexBufferSharedPtr vertexBuffer = mCustomVertexBufferRenderable->getHardwareVertexBuffer();
	mVertexBuffer = mCLRoot->getVertexBufferManager()->createVertexBuffer(vertexBuffer);
	mVertexBuffer->registerForCL();

	// set the args values 
	error  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) mVertexBuffer->getPointer());
	error |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &mesh_width);
	error |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &mesh_height);

	return true;
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	if (mIsOpenCLEnabled)
	{
		mTotalTime += deltaTime;
		cl_int ciErrNum;

		mVertexBuffer->map();

		// Set arg 3 and execute the kernel
		ciErrNum = clSetKernelArg(ckKernel, 3, sizeof(float), &mTotalTime);

		ciErrNum = clEnqueueNDRangeKernel(*mCLRoot->getCommandQueue(), ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0,0,0 );

		mVertexBuffer->unmap();

		clFinish(*mCLRoot->getCommandQueue());
	}

	bool result = processInputs(deltaTime);
	return result;
}

void OgreAppLogic::shutdown(void)
{
	mVertexBuffer->unregister();
	
	mCLRoot->shutdown();
	mCLRoot->getVertexBufferManager()->destroyVertexBuffer(mVertexBuffer);
	Ogre::OpenCL::Root::destroyRoot(mCLRoot);

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
		mIsOpenCLEnabled = !mIsOpenCLEnabled;
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