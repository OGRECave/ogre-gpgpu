#include "Kinect-win32.h"
#include "Kinect-win32-internal.h"

#include<algorithm>

namespace Kinect
{
	KinectFinder::KinectFinder()
	{
		usb_init();
		usb_find_busses();
		usb_find_devices();

		usb_bus *CurrentBus = usb_get_busses();

		std::vector< void *> KinectsFound;
		std::vector< void *> KinectMotorsFound;
		std::vector< void *> KinectAudioFound;

		while (CurrentBus)
		{
			usb_device_t * CurrentDev = CurrentBus->devices;
			while (CurrentDev)
			{
				if (CurrentDev->descriptor.idVendor == 0x045E &&
					CurrentDev->descriptor.idProduct == 0x02AE) // cam = 0x02AE  motor = 0x02B0  audio = 0x02AD
				{
					KinectsFound.push_back(CurrentDev);
				};
				if (CurrentDev->descriptor.idVendor == 0x045E &&
					CurrentDev->descriptor.idProduct == 0x02B0) // cam = 0x02AE  motor = 0x02B0  audio = 0x02AD
				{
					KinectMotorsFound.push_back(CurrentDev);
				};
				if (CurrentDev->descriptor.idVendor == 0x045E &&
					CurrentDev->descriptor.idProduct == 0x02AD) // cam = 0x02AE  motor = 0x02B0  audio = 0x02AD
				{
					KinectAudioFound.push_back(CurrentDev);
				};
				CurrentDev = CurrentDev->next;
			};
			CurrentBus = CurrentBus->next;
		};


		for (unsigned int i = 0;i<KinectsFound.size();i++)
		{
			void *Motor = NULL;
			if (i<KinectMotorsFound.size()) Motor = KinectMotorsFound[i];
			Kinect *K = new Kinect(KinectsFound[i], Motor);
			if (K->Opened())
			{
				mKinects.push_back(K);
			}
			else
			{
				delete K;
			}

		};


	};

	KinectFinder::~KinectFinder()
	{
		for (unsigned int i = 0;i<mKinects.size();i++)
		{
			delete mKinects[i];
		};
		mKinects.clear();
	};

	Kinect *KinectFinder::GetKinect(int index)
	{
		if (index>-1 && index< (int)mKinects.size()) return mKinects[index];
		return NULL;
	};

	int KinectFinder::GetKinectCount()
	{
		return mKinects.size();
	};

	bool Kinect::Opened()
	{
		KinectInternalData *KID = (KinectInternalData *) mInternalData;
		if (KID->mDeviceHandle) return true;
		return false;
	};

	Kinect::Kinect(void *internaldata, void *internalmotordata)
	{
		InitializeCriticalSection(&mListenersLock);
		KinectInternalData *KID = new KinectInternalData(this);
		mInternalData = (void *)KID;
		KID->OpenDevice((usb_device_t *)internaldata, (usb_device_t *)internalmotordata);

	};

	Kinect::~Kinect()
	{
		if (mInternalData) 
		{
			KinectInternalData *KID = (KinectInternalData *) mInternalData;
			delete KID;
		}
	};

	void Kinect::KinectDisconnected()
	{
		EnterCriticalSection(&mListenersLock);
		for (unsigned int i=0;i<mListeners.size();i++) mListeners[i]->KinectDisconnected(this);
		LeaveCriticalSection(&mListenersLock);
	};

	void Kinect::DepthReceived()
	{		
		EnterCriticalSection(&mListenersLock);
		for (unsigned int i=0;i<mListeners.size();i++) mListeners[i]->DepthReceived(this);
		LeaveCriticalSection(&mListenersLock);
	};

	void Kinect::ColorReceived()
	{	
		EnterCriticalSection(&mListenersLock);
		for (unsigned int i=0;i<mListeners.size();i++) mListeners[i]->ColorReceived(this);
		LeaveCriticalSection(&mListenersLock);
	};
	
	void Kinect::AddListener(KinectListener *KL)
	{
		EnterCriticalSection(&mListenersLock);
		if (KL) mListeners.push_back(KL);
		LeaveCriticalSection(&mListenersLock);
	};
	
	void Kinect::RemoveListener(KinectListener *KL)
	{
		EnterCriticalSection(&mListenersLock);
		std::vector<KinectListener*>::iterator f = find(mListeners.begin(), mListeners.end(), KL);
		if (f!= mListeners.end()) mListeners.erase(f);
		LeaveCriticalSection(&mListenersLock);
	};
	
	void Kinect::AudioReceived()
	{
		EnterCriticalSection(&mListenersLock);
		for (unsigned int i=0;i<mListeners.size();i++) mListeners[i]->AudioReceived(this);
		LeaveCriticalSection(&mListenersLock);
	};

	void Kinect::ParseColorBuffer()
	{
		KinectInternalData *KID = (KinectInternalData *) mInternalData;

		KID->LockRGB();
		for (int y=1; y<479; y++) 
		{
			for (int x=0; x<640; x++) 
			{
				int i = y*640+x;
				if (x&1) 
				{
					if (y&1) 
					{
						mColorBuffer[3*i+1] = KID->rgb_buf2[i];
						mColorBuffer[3*i+4] = KID->rgb_buf2[i];
					} 
					else 
					{
						mColorBuffer[3*i] = KID->rgb_buf2[i];
						mColorBuffer[3*i+3] = KID->rgb_buf2[i];
						mColorBuffer[3*(i-640)] = KID->rgb_buf2[i];
						mColorBuffer[3*(i-640)+3] = KID->rgb_buf2[i];
					}
				} 
				else 
				{
					if (y&1) 
					{
						mColorBuffer[3*i+2] = KID->rgb_buf2[i];
						mColorBuffer[3*i-1] = KID->rgb_buf2[i];
						mColorBuffer[3*(i+640)+2] = KID->rgb_buf2[i];
						mColorBuffer[3*(i+640)-1] = KID->rgb_buf2[i];
					}
					else 
					{
						mColorBuffer[3*i+1] = KID->rgb_buf2[i];
						mColorBuffer[3*i-2] = KID->rgb_buf2[i];
					}
				}
			}
		}
		KID->UnlockRGB();
	}
	
	void Kinect::ParseDepthBuffer()
	{
		KinectInternalData *KID = (KinectInternalData *) mInternalData;
		int bitshift = 0;
		KID->LockDepth();
		for (int i=0; i<640*480; i++) 
		{
			int idx = (i*11)/8;
			uint32_t word = (KID->depth_sourcebuf2[idx]<<16) | (KID->depth_sourcebuf2[idx+1]<<8) | KID->depth_sourcebuf2[idx+2];
			mDepthBuffer[i] = ((word >> (13-bitshift)) & 0x7ff);
			bitshift = (bitshift + 11) % 8;
		}
		KID->UnlockDepth();
	};

	void Kinect::SetMotorPosition(double newpos)
	{
		KinectInternalData *KID = (KinectInternalData *) mInternalData;
		KID->SetMotorPosition(newpos);
	};

	void Kinect::SetLedMode(int NewMode)
	{
		KinectInternalData *KID = (KinectInternalData *) mInternalData;
		KID->SetLedMode(NewMode);
	};


	bool Kinect::GetAcceleroData(float *x, float *y, float *z)
	{
		KinectInternalData *KID = (KinectInternalData *) mInternalData;
		if (x && y && z)
		{
			return KID->GetAcceleroData(x,y,z);
		};
		return false;
	};
};