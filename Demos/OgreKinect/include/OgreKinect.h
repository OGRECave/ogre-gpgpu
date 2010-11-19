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

#pragma once

#include <OgreKinectPrerequisites.h>
#include <OgreVector3.h>
#include <OgreTexture.h>
#include <Kinect-win32.h>

namespace Kinect
{
	class KinectFinder;
	class Kinect;
}

namespace Ogre
{
	namespace Kinect
	{
		enum LedMode
		{
			Led_Off                = 0x0,
			Led_Green              = 0x1,
			Led_Red                = 0x2,
			Led_Yellow             = 0x3,
			Led_BlinkingYellow     = 0x4,
			Led_BlinkingGreen      = 0x5,
			Led_AlternateRedYellow = 0x6,
			Led_AlternateRedGreen  = 0x7
		};

		static const unsigned int colorWidth        = ::Kinect::KINECT_COLOR_WIDTH;
		static const unsigned int colorHeight       = ::Kinect::KINECT_COLOR_HEIGHT;
		static const unsigned int depthWidth        = ::Kinect::KINECT_DEPTH_WIDTH;
		static const unsigned int depthHeight       = ::Kinect::KINECT_DEPTH_HEIGHT;
		static const unsigned int nbMicrophone      = ::Kinect::KINECT_MICROPHONE_COUNT;
		static const unsigned int audioBufferlength = ::Kinect::KINECT_AUDIO_BUFFER_LENGTH;

		class Device;
		class _OgreKinectExport DeviceManager
		{
			public:
				DeviceManager();
				virtual ~DeviceManager();

				unsigned int getKinectCount();
				Device* getKinect(unsigned int index = 0);

			protected:
				::Kinect::KinectFinder* mFinder;
		};

		class DeviceListener;
		class _OgreKinectExport Device
		{
			friend class DeviceListener;

			public:
				Device(::Kinect::Kinect* kinect);
				virtual ~Device();

				void createTexture(const std::string& colorTextureName, const std::string& depthTextureName);
				void createColoredDepthTexture(const std::string& coloredDepthTextureName);

				bool isConnected();
				bool update();

				void setMotorPosition(double value);
				void setLedMode(LedMode mode);
				bool getAccelerometerData(Ogre::Vector3& value);

			protected:												
				void convertDepthToRGB();

				::Kinect::Kinect* mKinect;
				DeviceListener*   mKinectListener;

				unsigned short    mGammaMap[2048];

				Ogre::TexturePtr mColorTexture;
				Ogre::PixelBox   mColorPixelBox;
				bool             mColorTextureAvailable;

				Ogre::TexturePtr mDepthTexture;
				Ogre::PixelBox   mDepthPixelBox;
				bool             mDepthTextureAvailable;

				Ogre::TexturePtr mColoredDepthTexture;
				Ogre::PixelBox   mColoredDepthPixelBox;
				unsigned char*   mColoredDepthBuffer;
		};

		class DeviceListener : public ::Kinect::KinectListener
		{
			public:
				DeviceListener(Device* device);

				virtual void KinectDisconnected(::Kinect::Kinect* kinect);
				virtual void ColorReceived(::Kinect::Kinect* kinect);
				virtual void DepthReceived(::Kinect::Kinect* kinect);				

			protected:
				Device* mDevice;
		};
	}
}