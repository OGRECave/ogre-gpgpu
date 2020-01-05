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