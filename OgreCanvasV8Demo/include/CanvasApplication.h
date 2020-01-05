#pragma once

#include "ExampleApplication.h"
#include "ExampleFrameListener.h"
#include "CanvasTexture.h"
#include "CanvasLogger.h"
#include "v8.h"
#include "DemoViewer.h"

class CanvasApplication : public ExampleApplication 
{
	public:
		CanvasApplication();
		void nextDemo();
		void previousDemo();

	protected:
		void setDemo(const Demo& _demo);
		void fillCanvas(const std::string& _scriptFilename);
		void emptyCanvas();

		void createScene(void);
		void destroyScene(void);
		void createFrameListener(void);

		void createCanvasMaterial(const std::string& _name, int _width, int _height);
		void deleteCanvasMaterial();

		int                        mDemoIndex;
		DemoViewer                 mDemoViewer;
		OgreCanvas::CanvasTexture* mTexture;
		OgreCanvas::CanvasContext* mContext;
		OgreCanvas::CanvasLogger*  mLogger;
		Ogre::Entity*              mEntity;
		Ogre::SceneNode*           mEntityNode;
};

class CanvasFrameListener : public ExampleFrameListener 
{
	public:
		CanvasFrameListener(Ogre::RenderWindow* _window, Ogre::Camera* _cam, CanvasApplication* _app);
		virtual bool frameStarted(const FrameEvent& evt);
		virtual bool processUnbufferedKeyInput(const Ogre::FrameEvent& evt);
	
	protected:
		CanvasApplication* mApp;
};