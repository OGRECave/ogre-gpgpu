#include <Ogre.h>


class StatsFrameListener : public Ogre::FrameListener
{
public:
	StatsFrameListener(Ogre::RenderTarget *window);
	~StatsFrameListener();
	virtual bool frameRenderingQueued(const Ogre::FrameEvent& evt);
	virtual bool frameEnded(const Ogre::FrameEvent& evt);

	void showDebugOverlay(bool show);
	void setDebugText(const Ogre::String &debugText) { mDebugText = debugText; }

protected:
	void updateStats(void);

	Ogre::String mDebugText;
	Ogre::Overlay* mDebugOverlay;
	Ogre::RenderTarget *mWindow;
	bool mIsAttached;
};
