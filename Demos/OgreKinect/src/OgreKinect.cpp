/*
	Copyright (c) 2010 ASTRE Henri (http://www.visual-experiments.com)

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

#include <OgreKinect.h>

#include <OgreTextureManager.h>
#include <OgreMaterialManager.h>
#include <OgreTechnique.h>
#include <OgreHardwarePixelBuffer.h>

using namespace Ogre::Kinect;

/*
	Device Manager
*/
DeviceManager::DeviceManager()
{
	mFinder = new ::Kinect::KinectFinder();
}

DeviceManager::~DeviceManager()
{
	delete mFinder;
}

unsigned int DeviceManager::getKinectCount()
{
	return (unsigned int) mFinder->GetKinectCount();
}

Device* DeviceManager::getKinect(unsigned int index)
{
	return new Device(mFinder->GetKinect(index));
}

/*
	Device
*/

Device::Device(::Kinect::Kinect* kinect)
{
	for (int i=0; i<2048; i++)
		mGammaMap[i] = (unsigned short)(float)(powf(i/2048.0f, 3)*6*6*256);

	mColorTexture.setNull();
	mDepthTexture.setNull();
	mColoredDepthTexture.setNull();
	mColorTextureAvailable = false;
	mDepthTextureAvailable = false;
	mColoredDepthBuffer    = NULL;

	mKinect         = kinect;
	mKinectListener = new DeviceListener(this);
	mKinect->AddListener(mKinectListener);
}

Device::~Device()
{
	delete mKinectListener;
	delete[] mColoredDepthBuffer;
}

bool Device::isConnected()
{
	return mKinect->Opened();
}

void Device::setMotorPosition(double value)
{
	mKinect->SetMotorPosition(value);
}

void Device::setLedMode(LedMode mode)
{
	mKinect->SetLedMode((int)mode);
}

bool Device::getAccelerometerData(Ogre::Vector3& value)
{
	return mKinect->GetAcceleroData(&value.x, &value.y, &value.z);
}

bool Device::update()
{
	bool updated = false;

	if (!mColorTexture.isNull() && mColorTextureAvailable)
	{
		Ogre::HardwarePixelBufferSharedPtr pixelBuffer = mColorTexture->getBuffer();
		pixelBuffer->blitFromMemory(mColorPixelBox);
		mColorTextureAvailable = false;
		updated = true;
	}
	if (!mDepthTexture.isNull() && mDepthTextureAvailable)
	{
		Ogre::HardwarePixelBufferSharedPtr pixelBuffer = mDepthTexture->getBuffer();
		pixelBuffer->blitFromMemory(mDepthPixelBox);

		if (!mColoredDepthTexture.isNull())
		{
			convertDepthToRGB();
			Ogre::HardwarePixelBufferSharedPtr pixelBuffer = mColoredDepthTexture->getBuffer();
			pixelBuffer->blitFromMemory(mColoredDepthPixelBox);		
		}
		mDepthTextureAvailable = false;
		updated = true;
	}

	return updated;
}

void Device::createTexture(const std::string& colorTextureName, const std::string& depthTextureName)
{
	mColorTexture = TextureManager::getSingleton().createManual(
		colorTextureName, 
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		TEX_TYPE_2D, 
		Ogre::Kinect::colorWidth, 
		Ogre::Kinect::colorHeight, 
		0,
		PF_R8G8B8, 
		TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

	mDepthTexture = TextureManager::getSingleton().createManual(
		depthTextureName, 
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		TEX_TYPE_2D, 
		Ogre::Kinect::depthWidth, 
		Ogre::Kinect::depthHeight, 
		0,
		PF_FLOAT32_R, 
		TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
}

void Device::createColoredDepthTexture(const std::string& coloredDepthTextureName)
{
	mColoredDepthTexture = TextureManager::getSingleton().createManual(
		coloredDepthTextureName, 
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		TEX_TYPE_2D, 
		Ogre::Kinect::depthWidth, 
		Ogre::Kinect::depthHeight, 
		0,
		PF_R8G8B8, 
		TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

	mColoredDepthBuffer   = new unsigned char[Ogre::Kinect::depthWidth * Ogre::Kinect::depthHeight * 3];
	mColoredDepthPixelBox = Ogre::PixelBox(Ogre::Kinect::depthWidth, Ogre::Kinect::depthHeight, 1, Ogre::PF_R8G8B8, mColoredDepthBuffer);
}

void Device::convertDepthToRGB()
{
	int i=0;
	for (int y=0; y<480; y++)
	{
		unsigned char* destrow = mColoredDepthBuffer + (y*(640))*3;
		for (int x=0; x<640; x++)
		{
			unsigned short Depth = mKinect->mDepthBuffer[i];
			int pval = mGammaMap[Depth];
			int lb = pval & 0xff;
			switch (pval>>8) 
			{
				case 0:
					destrow[2] = 255;
					destrow[1] = 255-lb;
					destrow[0] = 255-lb;
					break;
				case 1:
					destrow[2] = 255;
					destrow[1] = lb;
					destrow[0] = 0;
					break;
				case 2:
					destrow[2] = 255-lb;
					destrow[1] = 255;
					destrow[0] = 0;
					break;
				case 3:
					destrow[2] = 0;
					destrow[1] = 255;
					destrow[0] = lb;
					break;
				case 4:
					destrow[2] = 0;
					destrow[1] = 255-lb;
					destrow[0] = 255;
					break;
				case 5:
					destrow[2] = 0;
					destrow[1] = 0;
					destrow[0] = 255-lb;
					break;
				default:
					destrow[2] = 0;
					destrow[1] = 0;
					destrow[0] = 0;
					break;
			}
			destrow += 3;
			i++;
		}
	}
}

/*
	Device Listener
*/

DeviceListener::DeviceListener(Device* device)
{
	mDevice = device;
}

void DeviceListener::KinectDisconnected(::Kinect::Kinect* kinect)
{
	mDevice->mColorTextureAvailable = false;
	mDevice->mDepthTextureAvailable = false;
}

void DeviceListener::ColorReceived(::Kinect::Kinect* kinect)
{
	kinect->ParseColorBuffer();
	mDevice->mColorPixelBox = Ogre::PixelBox(Ogre::Kinect::colorWidth, Ogre::Kinect::colorHeight, 1, Ogre::PF_B8G8R8, kinect->mColorBuffer);
	mDevice->mColorTextureAvailable = true;
}

void DeviceListener::DepthReceived(::Kinect::Kinect* kinect)
{
	kinect->ParseDepthBuffer();
	mDevice->mDepthPixelBox = Ogre::PixelBox(Ogre::Kinect::depthWidth, Ogre::Kinect::depthHeight, 1, Ogre::PF_L16, kinect->mDepthBuffer);
	mDevice->mDepthTextureAvailable = true;	
}