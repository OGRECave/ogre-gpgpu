#ifndef KINECTWIN32INTERNAL
#define KINECTWIN32INTERNAL
#include "Kinect-win32.h"
#include "libusb\include\usb.h"
namespace Kinect
{
	enum
	{
//		DEPTH_LEN = 960,
//		RGB_LEN = 960,
		RGB_NUM_XFERS = 30,
		
		RGB_PKT_SIZE = 1920,
		RGB_PKTS_PER_XFER =16,

		RGB_XFER_SIZE = RGB_PKTS_PER_XFER*RGB_PKT_SIZE,

		DEPTH_NUM_XFERS = 10,

		DEPTH_PKT_SIZE = 1760,
		DEPTH_PKTS_PER_XFER =32,
		DEPTH_XFER_SIZE = DEPTH_PKTS_PER_XFER * DEPTH_PKT_SIZE 
	};


	typedef unsigned char uint8_t;
	typedef unsigned short uint16_t;
	typedef unsigned int uint32_t;

	struct frame_hdr {
		uint8_t magic[2];
		uint8_t pad;
		uint8_t flag;
		uint8_t unk1;
		uint8_t seq;
		uint8_t unk2;
		uint8_t unk3;
		uint32_t timestamp;
	};


	class KinectInternalData
	{
	public:
		KinectInternalData(Kinect *inParent);
		~KinectInternalData();

		void LockDepth(){EnterCriticalSection(&depth_lock);};
		void UnlockDepth(){LeaveCriticalSection(&depth_lock);};
		void LockRGB(){EnterCriticalSection(&rgb_lock);};
		void UnlockRGB(){LeaveCriticalSection(&rgb_lock);};
		
		void LockDepthThread(){EnterCriticalSection(&depththread_lock);};
		void UnlockDepthThread(){LeaveCriticalSection(&depththread_lock);};
		void LockRGBThread(){EnterCriticalSection(&rgbthread_lock);};
		void UnlockRGBThread(){LeaveCriticalSection(&rgbthread_lock);};

		void SetMotorPosition(double newpos);
		void SetLedMode(unsigned short NewMode);
		bool GetAcceleroData(float *x, float *y, float *z);


		int mErrorCount; 

		CRITICAL_SECTION rgb_lock;
		CRITICAL_SECTION depth_lock;
		CRITICAL_SECTION rgbthread_lock;
		CRITICAL_SECTION depththread_lock;
		
		int DepthPacketCount;
		int RGBPacketCount;

		usb_dev_handle *mDeviceHandle;
		usb_dev_handle *mDeviceHandle_Motor;
		usb_dev_handle *mDeviceHandle_Audio;
		Kinect *mParent;

		unsigned char depth_seq_init;
		unsigned char depth_seq;

		void OpenDevice(usb_device_t *dev,usb_device_t *motordev);

		void depth_process(uint8_t *buf, size_t len);
		void rgb_process(uint8_t *buf, size_t len);
		void cams_init();
		void send_init();

		uint8_t *depth_sourcebuf;
		uint8_t *depth_sourcebuf2;
		uint16_t *depth_frame;
		uint8_t *depth_frame_color;

		uint8_t *rgb_buf;
		uint8_t *rgb_buf2;
		uint8_t *rgb_frame;
		int depth_pos;
		int rgb_pos ;

		void *depth_xfers[DEPTH_NUM_XFERS];
		void *rgb_xfers[RGB_NUM_XFERS];

		unsigned char rgb_bufs[RGB_NUM_XFERS][RGB_XFER_SIZE*2];
		unsigned char depth_bufs[DEPTH_NUM_XFERS][DEPTH_XFER_SIZE*2];
		
		unsigned long mCurrentDepthContext;
		unsigned long mCurrentRGBContext;

		int depth_read();
		int rgb_read();
		void ReadBoth();

		bool Running;
		bool ThreadDone;
		
		void RunThread();
	};
};
#endif


