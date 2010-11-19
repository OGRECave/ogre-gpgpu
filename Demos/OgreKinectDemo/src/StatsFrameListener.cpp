#include "StatsFrameListener.h"		
#include <Ogre.h>

using namespace Ogre;

StatsFrameListener::StatsFrameListener(RenderTarget *window)
{
	mWindow = window;
	mDebugText = "";
	mDebugOverlay = OverlayManager::getSingleton().getByName("Core/DebugOverlay");
	mIsAttached = false;
	showDebugOverlay(true);
}


StatsFrameListener::~StatsFrameListener()
{
	showDebugOverlay(false);
}

bool StatsFrameListener::frameRenderingQueued(const FrameEvent& evt)
{
	return true;
}

bool StatsFrameListener::frameEnded(const FrameEvent& evt)
{
	if(mDebugOverlay)
		updateStats();
	return true;
}

void StatsFrameListener::showDebugOverlay(bool show)
{
	if (mDebugOverlay)
	{
		if (show)
		{
			if(!mIsAttached)
			{
				Ogre::Root::getSingleton().addFrameListener(this);
				mIsAttached = true;
			}
			mDebugOverlay->show();
		}
		else
		{
			if(mIsAttached)
			{
				Ogre::Root::getSingleton().removeFrameListener(this);
				mIsAttached = false;
			}
			mDebugOverlay->hide();
		}
	}
}

void StatsFrameListener::updateStats(void)
{
	static String currFps = "Current FPS: ";
	static String avgFps = "Average FPS: ";
	static String bestFps = "Best FPS: ";
	static String worstFps = "Worst FPS: ";
	static String tris = "Triangle Count: ";
	static String batches = "Batch Count: ";

	// update stats when necessary
	try {
		OverlayElement* guiAvg = OverlayManager::getSingleton().getOverlayElement("Core/AverageFps");
		OverlayElement* guiCurr = OverlayManager::getSingleton().getOverlayElement("Core/CurrFps");
		OverlayElement* guiBest = OverlayManager::getSingleton().getOverlayElement("Core/BestFps");
		OverlayElement* guiWorst = OverlayManager::getSingleton().getOverlayElement("Core/WorstFps");

		const RenderTarget::FrameStats& stats = mWindow->getStatistics();
		guiAvg->setCaption(avgFps + StringConverter::toString(stats.avgFPS));
		guiCurr->setCaption(currFps + StringConverter::toString(stats.lastFPS));
		guiBest->setCaption(bestFps + StringConverter::toString(stats.bestFPS)
			+" "+StringConverter::toString(stats.bestFrameTime)+" ms");
		guiWorst->setCaption(worstFps + StringConverter::toString(stats.worstFPS)
			+" "+StringConverter::toString(stats.worstFrameTime)+" ms");

		OverlayElement* guiTris = OverlayManager::getSingleton().getOverlayElement("Core/NumTris");
		guiTris->setCaption(tris + StringConverter::toString(stats.triangleCount));

		OverlayElement* guiBatches = OverlayManager::getSingleton().getOverlayElement("Core/NumBatches");
		guiBatches->setCaption(batches + StringConverter::toString(stats.batchCount));

		OverlayElement* guiDbg = OverlayManager::getSingleton().getOverlayElement("Core/DebugText");
		guiDbg->setCaption(mDebugText);
	}
	catch(...) { /* ignore */ }
}