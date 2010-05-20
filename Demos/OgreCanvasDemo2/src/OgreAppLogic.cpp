#include "OgreAppLogic.h"
#include "OgreApp.h"
#include <Ogre/Ogre.h>
#include <Ogre/OgrePanelOverlayElement.h>

#include "StatsFrameListener.h"

using namespace Ogre;

OgreAppLogic::OgreAppLogic() 
: mApplication(0), mChrono(false)
{
	// ogre
	mSceneMgr		= 0;
	mViewport		= 0;
	mStatsFrameListener  = 0;
	mCanvasTextureClock1 = 0;
	mCanvasTextureClock2 = 0;
	mCanvasTextureSun    = 0;
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

	mCanvasTextureClock1 = new Ogre::Canvas::Texture("CanvasClock1", 150, 150);
	mCanvasTextureClock1->createMaterial();

	mCanvasTextureClock2 = new Ogre::Canvas::Texture("CanvasClock2", 150, 150);
	mCanvasTextureClock2->createMaterial();

	mCanvasTextureSun = new Ogre::Canvas::Texture("CanvasSun", 300, 300);
	mCanvasTextureSun->createMaterial();

	mSun.load("canvas_sun.png",     "General");
	mEarth.load("canvas_earth.png", "General");
	mMoon.load("canvas_moon.png",   "General");

	createCanvasOverlay();
	createCanvasCube();
	mChrono.start();
	updateCanvas();

	return true;
}

void OgreAppLogic::createCanvasOverlay()
{
	Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
	Ogre::Overlay* overlay = overlayManager.create("Canvas/Overlay");

	Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "CanvasClock1/Panel"));
	panel->setMetricsMode(Ogre::GMM_PIXELS);
	panel->setMaterialName("CanvasClock1");
	panel->setDimensions(150.0f, 150.0f);
	panel->setPosition(0, 0);
	overlay->add2D(panel);

	panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "CanvasClock2/Panel"));
	panel->setMetricsMode(Ogre::GMM_PIXELS);
	panel->setMaterialName("CanvasClock2");
	panel->setDimensions(150.0f, 150.0f);
	panel->setPosition(mApplication->getRenderWindow()->getWidth()-150.0f, 0);
	overlay->add2D(panel);

	overlay->show();
}

void OgreAppLogic::createCanvasCube(void)
{
	Ogre::SceneNode* mClockNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	mClockNode->setPosition(-1.0, 0.5, 0);
	mClockNode->attachObject(createCubeMesh("ClockCube1", "CanvasClock1"));

					  mClockNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	mClockNode->setPosition(1.0, 0.5, 0);
	mClockNode->attachObject(createCubeMesh("ClockCube2", "CanvasClock2"));
}

void OgreAppLogic::updateCanvas()
{
	if (mChrono.getTimeElapsed() > 100)
	{
		updateClock1Canvas();
		updateClock2Canvas();
		//updateSunCanvas();
		mChrono.reset();
	}
}

bool OgreAppLogic::preUpdate(Ogre::Real deltaTime)
{
	return true;
}

bool OgreAppLogic::update(Ogre::Real deltaTime)
{
	bool result = processInputs(deltaTime);
	updateCanvas();
	return result;
}

void OgreAppLogic::shutdown(void)
{
	delete mCanvasTextureClock1;
	delete mCanvasTextureClock2;
	delete mCanvasTextureSun;

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

	mCamera->setPosition(0, 5, 5);
	mCamera->lookAt(0, 0.75f, 0.5);
	mCamera->setNearClipDistance(0.1f);
}

void OgreAppLogic::createScene(void)
{
	mSceneMgr->setAmbientLight(ColourValue(0.5,0.5,0.5));
	mSceneMgr->setSkyBox(true, "Examples/GridSkyBox");


	Ogre::Light* mLight =  mSceneMgr->createLight("SunLight");
	mLight->setPosition(Ogre::Vector3(150, 100, -150) );
}

//--------------------------------- update --------------------------------

bool OgreAppLogic::processInputs(Ogre::Real deltaTime)
{
	const Degree ROT_SCALE = Degree(40.0f);
	const Real MOVE_SCALE = 5.0f;
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

	rotX *= deltaTime;
	rotY *= deltaTime;
	translateVector *= deltaTime;

	// mouse moves
	const OIS::MouseState &ms = mouse->getMouseState();
	if (ms.buttonDown(OIS::MB_Right))
	{
		translateVector.x += ms.X.rel * 0.003f * MOVE_SCALE;	// Move camera horizontaly
		translateVector.y -= ms.Y.rel * 0.003f * MOVE_SCALE;	// Move camera verticaly
	}
	else
	{
		rotX += Degree(-ms.X.rel * 0.003f * ROT_SCALE);		// Rotate camera horizontaly
		rotY += Degree(-ms.Y.rel * 0.003f * ROT_SCALE);		// Rotate camera verticaly
	}	



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

void OgreAppLogic::updateClock1Canvas()
{
	Chrono chrono(true);
	Ogre::Canvas::Context* ctx = mCanvasTextureClock1->getContext();

	ctx->save();		
	ctx->clearRect(0, 0, 150, 150);
	ctx->strokeStyle(Ogre::ColourValue::Black);
	ctx->fillStyle(Ogre::ColourValue::White);
	ctx->fillRect(0, 0, 150, 150);
	ctx->translate(75, 75);
	ctx->scale(0.4f, 0.4f);
	ctx->rotate(-Ogre::Math::PI/2.0f);
	ctx->lineWidth(8);
	ctx->lineCap(Ogre::Canvas::LineCap_Round);

	// Hour marks
	ctx->save();
	for (float i=0; i<12; i++)
	{
		ctx->beginPath();
		ctx->rotate(Ogre::Math::PI/6.0f);
		ctx->moveTo(100, 0);
		ctx->lineTo(120, 0);
		ctx->stroke();
	}
	ctx->restore();

	// Minute marks
	ctx->save();
	ctx->lineWidth(5);
	for (int i=0; i<60; i++)
	{
		if (i%5!=0) 
		{
			ctx->beginPath();
			ctx->moveTo(117, 0);
			ctx->lineTo(120, 0);
			ctx->stroke();
		}
		ctx->rotate(Ogre::Math::PI/30);
	}
	ctx->restore();

	time_t now = time(NULL);
	struct tm * timeinfo;
	timeinfo = localtime(&now);
	int sec = timeinfo->tm_sec;
	int min = timeinfo->tm_min;
	int hr = timeinfo->tm_hour;
	hr = hr>=12 ? hr-12 : hr;

	ctx->fillStyle(Ogre::ColourValue::Black);

	// write Hours
	ctx->save();
	ctx->rotate( hr*(Ogre::Math::PI/6) + (Ogre::Math::PI/360)*min + (Ogre::Math::PI/21600)*sec );
	ctx->lineWidth(14);
	ctx->beginPath();
	ctx->moveTo(-20, 0);
	ctx->lineTo(80, 0);
	ctx->stroke();
	ctx->restore();

	// write Minutes
	ctx->save();
	ctx->rotate( (Ogre::Math::PI/30)*min + (Ogre::Math::PI/1800)*sec );
	ctx->lineWidth(10);
	ctx->beginPath();
	ctx->moveTo(-28, 0);
	ctx->lineTo(112, 0);
	ctx->stroke();
	ctx->restore();

	// Write seconds
	ctx->save();
	ctx->rotate(sec * Ogre::Math::PI/30);
	ctx->strokeStyle(Ogre::Canvas::ColourConverter::fromHexa("#D40000")); //red
	ctx->fillStyle(Ogre::Canvas::ColourConverter::fromHexa("#D40000")); //red
	ctx->lineWidth(6);
	ctx->beginPath();
	ctx->moveTo(-30, 0);
	ctx->lineTo(83, 0);
	ctx->stroke();
	ctx->beginPath();
	ctx->arc(0, 0, 10, 0, Ogre::Math::PI*2, true);
	ctx->fill();
	ctx->beginPath();
	ctx->arc(95, 0, 10, 0, Ogre::Math::PI*2, true);
	ctx->stroke();
	ctx->fillStyle(Ogre::Canvas::ColourConverter::fromHexa("#555")); //grey dark
	ctx->arc(0, 0, 3, 0, Ogre::Math::PI*2, true);
	ctx->fill();
	ctx->restore();

	ctx->beginPath();
	ctx->lineWidth(14);
	ctx->strokeStyle(Ogre::Canvas::ColourConverter::fromHexa("#325FA2")); //blue dark
	ctx->arc(0, 0, 142, 0, Ogre::Math::PI*2, true);
	ctx->stroke();

	ctx->restore();
	double renderTime = chrono.getTimeElapsed();
	mCanvasTextureClock1->uploadTexture();
	double uploadTime = chrono.getTimeElapsed()-renderTime;
	std::cout << "clock1 render: " << renderTime << "ms, upload: " << uploadTime << "ms" <<std::endl;
}

void OgreAppLogic::updateSunCanvas()
{
	Ogre::Canvas::Context* ctx = mCanvasTextureSun->getContext();

	//ctx->globalCompositeOperation(Ogre::Canvas::DrawingOperator_DestOver);
	ctx->clearRect(0, 0, 300, 300); // clear canvas  
	ctx->fillStyle(Ogre::ColourValue::Black);
	ctx->fillRect(0, 0, 300, 300);
	ctx->fillStyle(Ogre::ColourValue(0.0f, 0.0f, 0.0f, 0.4f));
	ctx->strokeStyle(Ogre::Canvas::ColourConverter::fromRGBA(0, 153, 255, 102));
	ctx->save();  
	ctx->translate(150, 150);  

	// Earth  
	time_t now = time(NULL);
	struct tm * timeinfo;
	timeinfo = localtime(&now);
	int sec = timeinfo->tm_sec;
	int min = timeinfo->tm_min;
	int hr = timeinfo->tm_hour;

	ctx->rotate( ((2*Ogre::Math::PI)/60)*sec);// + ((2*Math.PI)/60000)*time.getMilliseconds() );  
	ctx->translate(105,0);  
	ctx->fillRect(0,-12,50,24); // Shadow  
	ctx->drawImage(mEarth, -12.0f, -12.0f);  

	// Moon  
	ctx->save();  
	ctx->rotate( ((2*Ogre::Math::PI)/6)*sec);// + ((2*Math.PI)/6000)*time.getMilliseconds() );  
	ctx->translate(0.0f, 28.5f);  
	ctx->drawImage(mMoon, -3.5f, -3.5f);  
	ctx->restore();  

	ctx->restore();  

	ctx->beginPath();  
	ctx->arc(150, 150, 105, 0, Ogre::Math::PI*2, false); // Earth orbit  
	ctx->stroke();  

	ctx->drawImage(mSun, 0.0f, 0.0f, 300.0f, 300.0f); 

	mCanvasTextureSun->uploadTexture();
}

//Copied from BetaCairo demo : http://www.ogre3d.org/forums/viewtopic.php?t=26987 (created by betajaen)
Ogre::ManualObject* OgreAppLogic::createCubeMesh(const std::string& name, const std::string& matName)
{
	Ogre::ManualObject* cube = new Ogre::ManualObject(name);
	cube->begin(matName);

	cube->position(0.500000,-0.500000,1.000000);
	cube->normal(0.408248f,-0.816497f,0.408248f);
	cube->textureCoord(1,0);


	cube->position(-0.500000,-0.500000,0.000000);
	cube->normal(-0.408248f,-0.816497f,-0.408248f);
	cube->textureCoord(0,1);

	cube->position(0.500000,-0.500000,0.000000);
	cube->normal(0.666667f,-0.333333f,-0.666667f);
	cube->textureCoord(1,1);

	cube->position(-0.500000,-0.500000,1.000000);
	cube->normal(-0.666667f,-0.333333f,0.666667f);
	cube->textureCoord(0,0);

	cube->position(0.500000,0.500000,1.000000);
	cube->normal(0.666667f,0.333333f,0.666667f);
	cube->textureCoord(1,0);

	cube->position(-0.500000,-0.500000,1.000000);
	cube->normal(-0.666667f,-0.333333f,0.666667f);
	cube->textureCoord(0,1);

	cube->position(0.500000,-0.500000,1.000000);
	cube->normal(0.408248f,-0.816497f,0.408248f);
	cube->textureCoord(1,1);

	cube->position(-0.500000,0.500000,1.000000);
	cube->normal(-0.408248f,0.816497f,0.408248f);
	cube->textureCoord(0,0);

	cube->position(-0.500000,0.500000,0.000000);
	cube->normal(-0.666667f,0.333333f,-0.666667f);
	cube->textureCoord(0,1);

	cube->position(-0.500000,-0.500000,0.000000);
	cube->normal(-0.408248f,-0.816497f,-0.408248f);
	cube->textureCoord(1,1);

	cube->position(-0.500000,-0.500000,1.000000);
	cube->normal(-0.666667f,-0.333333f,0.666667f);
	cube->textureCoord(1,0);

	cube->position(0.500000,-0.500000,0.000000);
	cube->normal(0.666667f,-0.333333f,-0.666667f);
	cube->textureCoord(0,1);

	cube->position(0.500000,0.500000,0.000000);
	cube->normal(0.408248f,0.816497f,-0.408248f);
	cube->textureCoord(1,1);

	cube->position(0.500000,-0.500000,1.000000);
	cube->normal(0.408248f,-0.816497f,0.408248f);
	cube->textureCoord(0,0);

	cube->position(0.500000,-0.500000,0.000000);
	cube->normal(0.666667f,-0.333333f,-0.666667f);
	cube->textureCoord(1,0);

	cube->position(-0.500000,-0.500000,0.000000);
	cube->normal(-0.408248f,-0.816497f,-0.408248f);
	cube->textureCoord(0,0);

	cube->position(-0.500000,0.500000,1.000000);
	cube->normal(-0.408248f,0.816497f,0.408248f);
	cube->textureCoord(1,0);

	cube->position(0.500000,0.500000,0.000000);
	cube->normal(0.408248f,0.816497f,-0.408248f);
	cube->textureCoord(0,1);

	cube->position(-0.500000,0.500000,0.000000);
	cube->normal(-0.666667f,0.333333f,-0.666667f);
	cube->textureCoord(1,1);

	cube->position(0.500000,0.500000,1.000000);
	cube->normal(0.666667f,0.333333f,0.666667f);
	cube->textureCoord(0,0);

	cube->triangle(0,1,2);
	cube->triangle(3,1,0);
	cube->triangle(4,5,6);
	cube->triangle(4,7,5);
	cube->triangle(8,9,10);
	cube->triangle(10,7,8);
	cube->triangle(4,11,12);
	cube->triangle(4,13,11);
	cube->triangle(14,8,12);
	cube->triangle(14,15,8);
	cube->triangle(16,17,18);
	cube->triangle(16,19,17);
	cube->end(); 

	return cube;

}

//Adapted from BetaCairo demo : http://www.ogre3d.org/forums/viewtopic.php?t=26987 (created by betajaen)
void OgreAppLogic::updateClock2Canvas()
{
	Chrono chrono(true);

	time_t now = time(NULL);
	struct tm * timeinfo;
	timeinfo = localtime(&now);
	float sec  = (float) timeinfo->tm_sec;
	float min  = (float) timeinfo->tm_min;
	float hour = (float) timeinfo->tm_hour;

	float clockSize = 150.0f;
	float clockUpdate = 0.125f;

	Ogre::Canvas::Context* ctx = mCanvasTextureClock2->getContext();
	Ogre::ColourValue tomato     = Ogre::Canvas::ColourConverter::fromHexa("#ff6347");
	Ogre::ColourValue whitesmoke = Ogre::Canvas::ColourConverter::fromHexa("#f5f5f5");

	// Basic bits:

		// Paint the Drawing with a tomatoes, this pretty much wipes over the previous drawing.
		ctx->fillStyle(tomato);
		ctx->fillRect(0, 0, clockSize, clockSize);

		// Draw a outline around the texture, using the whitesmoke colour and a line thickness of 1 pixel
		ctx->strokeStyle(whitesmoke);
		ctx->lineWidth(1.0f);
		ctx->strokeRect(0, 0, clockSize, clockSize);

		// Path out an arc (semi or full circle)
		ctx->fillStyle(whitesmoke);
		ctx->arc(clockSize / 2.0f, clockSize / 2.0f, clockSize / 4.0f, 0.0f, 360.0f, false);
		ctx->fill();

	// The Hour hand:

		ctx->lineCap(Ogre::Canvas::LineCap_Round);
		ctx->strokeStyle(tomato);
		ctx->lineWidth(3.0f);
		ctx->moveTo(clockSize / 2.0f, clockSize / 2.0f); // Move to the center of the image

		// Using a simple bit of trig. path the hour hand from the center to the correct hour, using a length of 1/5th of the imagesize.
		ctx->lineTo((clockSize / 2.0f) + (Ogre::Math::Sin(hour * Ogre::Math::PI / 6.0f) * (clockSize / 5.0f)), (clockSize / 2.0f) + (-Ogre::Math::Cos(hour * Ogre::Math::PI / 6.0f) * (clockSize / 5.0f)));
		ctx->stroke();


	// The Minute hand:
		
		ctx->beginPath();
		ctx->moveTo(clockSize / 2.0f, clockSize / 2.0f); // Move to the center of the image

		// Now draw the hour hand, using a length of 1/6th of the imagesize.
		ctx->lineTo((clockSize / 2) + (Ogre::Math::Sin((min + (sec / 60.0f)) * Ogre::Math::PI / 30.0f) * (clockSize / 6.0f)), (clockSize / 2) + (-Ogre::Math::Cos((min + (sec / 60.0f)) * Ogre::Math::PI / 30.0f) * (clockSize / 6.0f)));
		ctx->stroke();
		ctx->lineWidth(2.0f);
		ctx->strokeStyle(whitesmoke);
		ctx->stroke();

		// The second hand, and previous seconds:

			ctx->lineWidth(1.5f);		
			Ogre::ColourValue c = whitesmoke;

			int lastSecond = (int) Ogre::Math::Ceil(sec);
			float a = 1.0f / sec;

			// For each previous second, draw it out, with a subtle fade (alpha). The alpha decreases based how close it is to the current second.
			for (int i=0; i<lastSecond; i++) 
			{
				ctx->strokeStyle(Ogre::ColourValue(c.r, c.g, c.b, a*i));
				ctx->moveTo((clockSize / 2)  + (Ogre::Math::Sin(i * Ogre::Math::PI / 30.0f) * (clockSize / 4.0f) ),  (clockSize / 2.0f)  + (-Ogre::Math::Cos(i * Ogre::Math::PI / 30.0f) * (clockSize / 4.0f)));
				ctx->lineTo((clockSize / 2)  + (Ogre::Math::Sin(i * Ogre::Math::PI / 30.0f) * (clockSize / 3.5f) ) , (clockSize / 2.0f)  + (-Ogre::Math::Cos(i * Ogre::Math::PI / 30.0f) * (clockSize / 3.5f)));
				ctx->stroke();
			}

		// Draw the main second.

			ctx->strokeStyle(whitesmoke);
			ctx->moveTo((clockSize / 2.0f)  + (Ogre::Math::Sin(sec * Ogre::Math::PI / 30.0f) * (clockSize / 4.0f) ),  (clockSize / 2.0f)  + (-Ogre::Math::Cos(sec * Ogre::Math::PI / 30.0f) * (clockSize / 4.0f)));
			ctx->lineTo((clockSize / 2.0f)  + (Ogre::Math::Sin(sec * Ogre::Math::PI / 30.0f) * (clockSize / 3.5f) ) , (clockSize / 2.0f)  + (-Ogre::Math::Cos(sec * Ogre::Math::PI / 30.0f) * (clockSize / 3.5f)));
			ctx->stroke();

		// Outer ring:

			ctx->strokeStyle(whitesmoke);		
			ctx->arc((clockSize / 2.0f),(clockSize / 2.0f), (clockSize / 3.5f), 0.0f, 360.0f, false); // Path out an arc, from the center, for a radius of 1/3.5th of the image
			ctx->stroke();

		// The 12:

			// Path and Stroke "12"
			ctx->fillText("12", (clockSize / 2.0f)-8, (clockSize / 5.0f));

	double renderTime = chrono.getTimeElapsed();
	mCanvasTextureClock2->uploadTexture();
	double uploadTime = chrono.getTimeElapsed()-renderTime;
	std::cout << "clock2 render: " << renderTime << "ms, upload: " << uploadTime << "ms" <<std::endl;
}