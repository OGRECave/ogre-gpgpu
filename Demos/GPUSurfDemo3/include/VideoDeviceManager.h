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

#include <string>
#include <vector>

#include <Ogre.h>

#include "videoInput.h"


/*
	//Example 1
	VideoDeviceManager vdm;
	if (vdm.size() > 0)
	{
		VideoDevice* device = vdm[0];
		device->init(640, 480, 60);

		device->showControlPanel();

		device->update(); //on app loop

		device->shutdown();
	}

	//Example 2
	VideoDeviceManager vdm;
	cout << vdm.size() << " device(s) found"<<endl;
	for (unsigned int i=0; i<vdm.size(); ++i)
	{
		VideoDevice* device = vdm[i];
		cout << "["<<i<<"] " <<device->getName()<<endl;
	}
*/

class VideoDeviceManager;

class VideoDevice
{
	public:			
		bool init(int width = 640, int height = 480, int fps = 60);
		bool update();
		void shutdown();

		void showControlPanel();

		std::string getName() const;
		int getWidth() const;
		int getHeight() const;
		size_t getBufferSize() const;
		void* getBufferData() const;

		void createTexture(const std::string name);

	protected:
		VideoDevice(VideoDeviceManager* manager, int index);
		~VideoDevice();

		VideoDeviceManager* mManager;
		int mIndex;
		bool mIsWorking;

		std::string mName;
		int mWidth;
		int mHeight;
		size_t mBufferSize;
		
		unsigned char* mBuffer;

		Ogre::TexturePtr mTexture;
		Ogre::PixelBox mPixelBox;

	friend class VideoDeviceManager;
};

class VideoDeviceManager
{
	public:
		VideoDeviceManager();
		~VideoDeviceManager();
		
		unsigned int size() const;
		VideoDevice* operator[](int index);

	protected:
		videoInput* mVideoInput;
		std::vector<VideoDevice*> mDevices;

	friend class VideoDevice;
};