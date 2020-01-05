#include "VideoDeviceManager.h"

#include "videoInput.h"

#include <OgreTextureManager.h>
#include <OgreMaterialManager.h>
#include <OgreTechnique.h>

using namespace Ogre;

/******* VIDEO DEVICE *******/

VideoDevice::VideoDevice(VideoDeviceManager* manager, int index)
{
	mManager   = manager;
	mIndex     = index;
	mIsWorking = false;

	mName       = mManager->mVideoInput->getDeviceName(mIndex);
	mWidth      = -1;
	mHeight     = -1;
	mBufferSize = -1;

	mBuffer = NULL;
	mTexture.setNull();
}

VideoDevice::~VideoDevice()
{
	delete[] mBuffer;
	mBuffer = NULL;
}

bool VideoDevice::init(int width, int height, int fps)
{
	mManager->mVideoInput->setIdealFramerate(mIndex, fps);
	mManager->mVideoInput->setupDevice(mIndex, width, height);	

	mWidth      = mManager->mVideoInput->getWidth(mIndex);
	mHeight     = mManager->mVideoInput->getHeight(mIndex);
	mBufferSize = mManager->mVideoInput->getSize(mIndex);

	delete[] mBuffer;
	mBuffer = new unsigned char[mBufferSize];

	mPixelBox = Ogre::PixelBox(mWidth, mHeight, 1, Ogre::PF_B8G8R8, mBuffer);

	mIsWorking = mManager->mVideoInput->isDeviceSetup(mIndex);

	return mIsWorking;
}

bool VideoDevice::update()
{
	bool updated = false;
	if (mIsWorking && mManager->mVideoInput->isFrameNew(mIndex))
	{
		mManager->mVideoInput->getPixels(mIndex, mBuffer, true, true);
		
		if (!mTexture.isNull())
		{
			HardwarePixelBufferSharedPtr pixelBuffer = mTexture->getBuffer();
			pixelBuffer->blitFromMemory(mPixelBox);
		}

		updated = true;
	}
	return updated;
}

void VideoDevice::shutdown()
{
	if (mIsWorking)
		mManager->mVideoInput->stopDevice(mIndex);
	mIsWorking = false;
	mTexture.setNull();
}

void VideoDevice::showControlPanel()
{
	mManager->mVideoInput->showSettingsWindow(mIndex);
}

std::string VideoDevice::getName() const
{
	return mName;
}

int VideoDevice::getWidth() const
{
	return mWidth;
}

int VideoDevice:: getHeight() const
{
	return mHeight;
}

size_t VideoDevice::getBufferSize() const
{
	return mBufferSize;
}

void* VideoDevice::getBufferData() const
{
	return mBuffer;
}

void VideoDevice::createTexture(const std::string name)
{
	mTexture = TextureManager::getSingleton().createManual(
		name, 
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		TEX_TYPE_2D, 
		mWidth, 
		mHeight, 
		0,
		PF_R8G8B8, 
		TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
}

/******* VIDEO DEVICE MANAGER *******/

VideoDeviceManager::VideoDeviceManager()
{
	videoInput::setVerbose(false);
	mVideoInput = new videoInput;
	mVideoInput->setUseCallback(true); //async
	
	int nbDevice = mVideoInput->listDevices(true);
	for (int i=0; i<nbDevice; ++i)
		mDevices.push_back(new VideoDevice(this, i));
}

VideoDeviceManager::~VideoDeviceManager()
{
	for (unsigned int i=0; i<mDevices.size(); ++i)
	{
		mDevices[i]->shutdown();
		delete mDevices[i];
	}

	delete mVideoInput;
}

unsigned int VideoDeviceManager::size() const
{
	return mDevices.size();
}

VideoDevice* VideoDeviceManager::operator[](int index)
{
	return mDevices[index];
}