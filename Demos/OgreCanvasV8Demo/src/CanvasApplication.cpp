#include "CanvasApplication.h"
#include "CanvasTexture.h"
#include "CanvasContextV8Bindings.h"

using namespace Ogre;
using namespace OgreCanvas;

CanvasFrameListener::CanvasFrameListener(Ogre::RenderWindow* window, Ogre::Camera* maincam, CanvasApplication* _app) 
: ExampleFrameListener(window, maincam), mApp(_app)
{
	showDebugOverlay(true);
}

bool CanvasFrameListener::frameStarted(const FrameEvent& evt)
{
	return true;
}

bool CanvasFrameListener::processUnbufferedKeyInput(const  Ogre::FrameEvent& evt)
{
	if( mKeyboard->isKeyDown(OIS::KC_ADD) && mTimeUntilNextToggle <= 0 )
	{
		mApp->nextDemo();
		mTimeUntilNextToggle = 0.2;
	}
	else if( mKeyboard->isKeyDown(OIS::KC_SUBTRACT) && mTimeUntilNextToggle <= 0 )
	{
		mApp->previousDemo();
		mTimeUntilNextToggle = 0.2;
	}

	return ExampleFrameListener::processUnbufferedKeyInput(evt);
}

CanvasApplication::CanvasApplication()
{
	mDemoIndex  = 0;
	mTexture    = NULL;
	mContext    = NULL;
	mEntity     = NULL;
	mEntityNode = NULL;
	mLogger     = new OgreCanvas::CanvasLogger("canvasLog.txt");
}

void CanvasApplication::createScene(void) 
{
    mSceneMgr->setAmbientLight(Ogre::ColourValue(0.2, 0.2, 0.2));
	mSceneMgr->setSkyBox(true, "Examples/OutDoorCubeMap", 50);
    mSceneMgr->setFog(Ogre::FOG_LINEAR, Ogre::ColourValue::Black, 1, 600,760);
	mSceneMgr->setAmbientLight(Ogre::ColourValue(0.6,0.6,0.6));
	mSceneMgr->setShadowTechnique(Ogre::SHADOWTYPE_STENCIL_ADDITIVE);

    Ogre::Light* mLight =  mSceneMgr->createLight("SunLight");
	mLight->setPosition(Ogre::Vector3(150, 100, -150) );

    mCamera->setPosition(0, 3, 0);
	mCamera->lookAt(0,0.75f,0.5);
	mCamera->setNearClipDistance(0.1f);
	
	mEntity = mSceneMgr->createEntity("CubeEntity", "quad1x1.mesh");	
	mEntity->setMaterialName("Examples/10PointBlock");

	mEntityNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("CubeNode");
	mEntityNode->attachObject(mEntity);

	if (mDemoViewer.getNbDemo() > 0)
		setDemo(mDemoViewer.getDemo(0));
}

void CanvasApplication::destroyScene(void)
{
	delete mTexture;
}

void CanvasApplication::createFrameListener(void) 
{
    mFrameListener = new CanvasFrameListener(mWindow, mCamera, this);
    mRoot->addFrameListener(mFrameListener);
}

void CanvasApplication::createCanvasMaterial(const std::string& _name, int _width, int _height)
{
	if (mTexture != NULL)
		deleteCanvasMaterial();

	mTexture = new OgreCanvas::CanvasTexture(_name, _width, _height, true);
	mContext = mTexture->getContext();

	mTexture->uploadTexture();
	mTexture->createMaterial();
}

void CanvasApplication::deleteCanvasMaterial()
{
	mTexture->deleteMaterial();
	delete mTexture;
	mTexture = NULL;
	mContext = NULL;
}

void CanvasApplication::fillCanvas(const std::string& _scriptFilename)
{
	CanvasContextV8Bindings canvasJS;
	canvasJS.loadScript(_scriptFilename, mContext, mLogger);

	mTexture->uploadTexture();
}

void CanvasApplication::emptyCanvas()
{
	OgreCanvas::CanvasContext* ctx = mContext;
	ctx->save();
	ctx->fillStyle(Ogre::ColourValue::White);
	ctx->fillRect(0, 0, 150, 200);
	ctx->fillStyle(Ogre::ColourValue::Black);
	ctx->restore();
	mTexture->uploadTexture();
}

void CanvasApplication::nextDemo()
{
	mDemoIndex++;
	if (mDemoIndex >= mDemoViewer.getNbDemo())
		mDemoIndex = 0;
	
	setDemo(mDemoViewer.getDemo(mDemoIndex));
}

void CanvasApplication::previousDemo()
{
	mDemoIndex--;
	if (mDemoIndex < 0)
		mDemoIndex = mDemoViewer.getNbDemo()-1;

	setDemo(mDemoViewer.getDemo(mDemoIndex));
}

void CanvasApplication::setDemo(const Demo& _demo)
{
	mEntity->setMaterialName("Examples/10PointBlock");
	createCanvasMaterial("CanvasDemo", _demo.width, _demo.height);
	mEntityNode->setScale(_demo.width, _demo.height, 1);
	mEntity->setMaterialName("CanvasDemo");
	emptyCanvas();
	fillCanvas(_demo.script);
}