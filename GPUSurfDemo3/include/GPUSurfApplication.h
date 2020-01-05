#pragma once

#include "Ogre.h"
#include "DShowVideo.h"
#include "DShowTextureSource.h"
#include "ExampleApplication.h"
#include "ExampleFrameListener.h"
#include "GPUSurf.h"

class GPUSurfApplication : public ExampleApplication
{
	public:
		GPUSurfApplication();
		virtual ~GPUSurfApplication();

		void nextStage();
		void prevStage();
		
		Ogre::Camera* getCamera() { return mCamera; }
		Ogre::SceneManager* getSceneManager() { return mSceneMgr; }

	protected:
		void createScene(void);
		void destroyScene(void);
		void createFrameListener(void);

		void createCamera(void);
		void createWebcamMaterial();
		void createWebcamPlane(Ogre::Real _distanceFromCamera);

		int mStageNumber;
		int mFrameNumber;
		
		Ogre::Rectangle2D* mMiniScreen;

		Ogre::DShowVideo*  mVideo;
		Ogre::SceneNode*   mCameraNode;
		GPUSurf*           mGPUSurf;
};

class GPUSurfFrameListener : public ExampleFrameListener 
{
	public:
		GPUSurfFrameListener(Ogre::RenderWindow* _window, Ogre::Camera* _cam, GPUSurfApplication* _app);
		bool processUnbufferedKeyInput(const Ogre::FrameEvent& evt);
	
	protected:
		GPUSurfApplication* mApp;
};