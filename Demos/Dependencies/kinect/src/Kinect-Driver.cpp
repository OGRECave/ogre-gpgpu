/*
 * This file was ported by Stijn Kuipers / Zephod from a part of the OpenKinect 
 * Project. http://www.openkinect.org
 *
 * Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#include <conio.h>

#include "Kinect-win32.h"
#include "Kinect-win32-internal.h"

#define DORGB 1
#define DODEPTH 1

namespace Kinect
{
	#include "init.h"

	DWORD WINAPI DepthThread( LPVOID lpParam ) 
	{ 
		KinectInternalData *KID  = (KinectInternalData*) lpParam;
		if (DODEPTH)
		{
		}
		else
		{
			return 0;
		};

		KID->LockRGBThread();
		
		for (int i=0; i<DEPTH_NUM_XFERS; i++) 
		{
			int ret;
			if (DODEPTH)
			{
				KID->depth_xfers[i] = 0;
				ret = usb_isochronous_setup_async(KID->mDeviceHandle, &KID->depth_xfers[i], 0x82, 1760);
				if (ret<0) 
				{
					printf("error setting up isochronous request for depth_xfer[%d]!", i);	
					KID->mErrorCount++;
				};
			};
		}


		for (int i=0; i<DEPTH_NUM_XFERS; i++) 
		{
			int ret;
			if (DODEPTH)
			{
				ZeroMemory(KID->depth_bufs[i], DEPTH_XFER_SIZE);
				ret = usb_submit_async(KID->depth_xfers[i], (char*)KID->depth_bufs[i], DEPTH_XFER_SIZE);
				if (ret<0)
				{
					printf("error submitting request!");
					KID->mErrorCount++;
				};
			};
		}
		while (KID->Running)
		{
			int loopcount = 0;
			while (	KID->depth_read() && loopcount++ < 20);
			//if(loopcount == 20) printf("too much depth");
			Sleep(1);

		};
		KID->UnlockRGBThread();
		
		return 0;
	};


	DWORD WINAPI RGBThread( LPVOID lpParam ) 
	{ 
		KinectInternalData *KID  = (KinectInternalData*) lpParam;
		KID->LockDepthThread();
		
		for (int i=0; i<RGB_NUM_XFERS; i++) 
		{
			int ret;
			if (DORGB)
			{
				KID->rgb_xfers[i] = 0;
				ret = usb_isochronous_setup_async(KID->mDeviceHandle, &KID->rgb_xfers[i], 0x81, RGB_PKT_SIZE);
				if (ret<0)
				{
					printf("error setting up isochronous request for rgb_xfer[%d]!", i);
					KID->mErrorCount++;
				}
			};
		};

		for (int i=0; i<RGB_NUM_XFERS; i++) 
		{
			int ret;
			if (DORGB)
			{
				ZeroMemory(&KID->rgb_bufs[i][0], RGB_XFER_SIZE);
				ret = usb_submit_async(KID->rgb_xfers[i], (char*)&KID->rgb_bufs[i][0], RGB_XFER_SIZE);
				if (ret<0)
				{
					printf("error submitting request!");
					KID->mErrorCount++;
				};
			};
		};

		while (KID->Running)
		{
			int loopcount = 0;
			while (	KID->rgb_read() && loopcount++ < 20);
			//if(loopcount == 20) printf("too much rgb");
			Sleep(0);
		};
		KID->UnlockDepthThread();
		
		return 0;
	};

	void  KinectInternalData::depth_process(uint8_t *buf, size_t len)
	{
		if (len < sizeof(frame_hdr))
		{
			return;
		};

		frame_hdr *hdr = (frame_hdr *)buf;
		uint8_t *data = buf + sizeof(frame_hdr);
		int datalen = len - sizeof(frame_hdr);

		if (!(hdr->magic[0] == 0x52 && hdr->magic[1] == 0x42))
		{
			return;
		};

		switch (hdr->flag) 
		{
		case 0x71:
			depth_pos = 0;
			DepthPacketCount = 0;
			depth_seq_init = hdr->seq;
		case 0x72:
		case 0x75:
			{
				unsigned char pos = hdr->seq - depth_seq_init;			
				memcpy(&depth_sourcebuf[depth_pos], data, __min(datalen, 1760-sizeof(frame_hdr)));
				depth_pos+=datalen;
				DepthPacketCount++;
			}
			break;
		}

		unsigned char newint = depth_seq + 1;
		if (newint != hdr->seq)
		{
			printf("seq lost! %02x %02x \n", newint, hdr->seq);
			depth_seq = hdr->seq;
			//return 1760;
		}
		else
		{
			//printf("in seq! %02x\n", newint);
		};

		depth_seq = hdr->seq;
		//if (depth_pos < 640*480*2) return 1760;

		if (hdr->flag != 0x75) return;

//		printf("packetcount: %d\n",packetcount);
		if (DepthPacketCount == 0xf2)
		{
			LockDepth();
			memcpy(depth_sourcebuf2, depth_sourcebuf, 640*480*2);
			UnlockDepth();
		
			mParent->DepthReceived();
		}
		else
		{
			printf("depth frame dropped - too incomplete!\n");
		};
		return;	
	}

	void KinectInternalData::rgb_process(uint8_t *buf, size_t len)
	{	
		if (len < sizeof(frame_hdr)) return;
		
		frame_hdr *hdr = (frame_hdr *)buf;
		uint8_t *data = buf + sizeof(frame_hdr);
		int datalen = len - sizeof(frame_hdr);

		if (!(hdr->magic[0] == 0x52 && hdr->magic[1] == 0x42))
		{
		 return;
		}

		switch (hdr->flag) 
		{
		case 0x81:
			rgb_pos = 0;
			RGBPacketCount = 0;
		case 0x82:
		case 0x85:
			memcpy(&rgb_buf[rgb_pos], data, datalen);
			rgb_pos += datalen;
			RGBPacketCount++;
			break;
		}

		if (hdr->flag != 0x85)
			return;
		if (RGBPacketCount > 0xa1)
		{
			//printf("GOT RGB FRAME, %d bytes\n", rgb_pos);
			
			LockRGB();
			memcpy(rgb_buf2,rgb_buf, 640*480*3);
			UnlockRGB();

			mParent->ColorReceived();
		}
		else
		{
			printf("rgb frame dropped - too incomplete!\n");
		}
	}



//	static int depthcount = 0;
//	static int depthbytes = 0;
//	static int rgbcount = 0;
//	static int rgbbytes = 0;

	int KinectInternalData::depth_read()
	{
		int i = 1;
		int retval = 0;
		i = usb_reap_async_nocancel(depth_xfers[mCurrentDepthContext], 10000);
		if (i>0)
		{
			retval = i;
			int off = 0;
			int P = DEPTH_PKT_SIZE;
			i = DEPTH_XFER_SIZE;
			while (i >0)
			{
				int len = __min(i,P);
				depth_process(&depth_bufs[mCurrentDepthContext][off], DEPTH_PKT_SIZE);
				int used = DEPTH_PKT_SIZE;
				off += used;
				i-= used;
			};
			if (i > 0)
			{
				printf("%d bytes in last packet\n", P- -i);
			}
		} 
		else
		{
			if (i<0)
			{
				if (i == -116 )
				{
					// timeout = not ready yet!
					return 0;
				}
				printf("rgb code %d...\n",i);
				usb_cancel_async(depth_xfers[mCurrentDepthContext]);
			}
			else
			{
				//	printf(",");
			};
		};
		ZeroMemory(depth_bufs[mCurrentDepthContext], DEPTH_XFER_SIZE);
		int ret = usb_submit_async(depth_xfers[mCurrentDepthContext], (char*)depth_bufs[mCurrentDepthContext],  DEPTH_XFER_SIZE);
		if( ret < 0 )
		{
			printf("error: %s\n", usb_strerror());
			usb_cancel_async(depth_xfers[mCurrentDepthContext]);
		}
		mCurrentDepthContext = (mCurrentDepthContext + 1) % DEPTH_NUM_XFERS;
		return retval;
	}

	void KinectInternalData::ReadBoth()
	{
		int retbytes = 1;
		int loopcount = 0;
		while (retbytes >0 && loopcount<20)
		{
			retbytes = 0;
			if (DORGB) retbytes += rgb_read();
			if (DODEPTH) retbytes += depth_read();
			loopcount++;
		};
		if (loopcount == 20) printf("too many pending reads.. breaking out\n");
	};

	int KinectInternalData::rgb_read()
	{
		int i = 1;
		int retval = 0;
		i = usb_reap_async_nocancel(rgb_xfers[mCurrentRGBContext], 10000);
		if (i>0)
		{
			retval = i;
			//rgbcount++ ;
			//rgbbytes+=i;
			//if (rgbcount%200== 0) printf("rgbbytes: %.3fMiB\n", (rgbbytes/1024.0)/1024.0);
			int off = 0;
			int P = RGB_PKT_SIZE;
			i = RGB_XFER_SIZE;
			int j = 0;
			bool magicfound = false;
			while (j<32 && magicfound == false)
			{
				if (rgb_bufs[mCurrentRGBContext][(j*RGB_PKT_SIZE)]  == 'R'
					&& rgb_bufs[mCurrentRGBContext][(j*RGB_PKT_SIZE) +1] == 'B')
				{
					magicfound = true;
				}
				else
				{
					j++;
				}
				
			}
			
			if (j>0)
			{
				off = j*RGB_PKT_SIZE;
				i-=off;
				//printf("skipped %d rgb packets in context %d!\n",j, mCurrentRGBContext);
			};
			int used = 0;
			while (i >0)
			{
				int len = __min(i,P);
				if (i<RGB_PKT_SIZE)
				{
					if (i == 960)
					{
					//	printf("last packet is halve!");
						if (rgb_bufs[mCurrentRGBContext][off]  == 'R'
					&& rgb_bufs[mCurrentRGBContext][off +1] == 'B')
						{
							printf ("valid halve packet found!\n");
						}
					};
				}

				if (rgb_bufs[mCurrentRGBContext][off]  == 'R' && rgb_bufs[mCurrentRGBContext][off +1] == 'B')
				{
					rgb_process(&rgb_bufs[mCurrentRGBContext][off], __min(i,RGB_PKT_SIZE));
					used = __min(i,RGB_PKT_SIZE);
				}
				else
				{
					used = 960;
				};
				off += used;
				i-= used;
			};			
		} 
		else
		{
			if (i<0)
			{
				if (i == -116 )
				{
					// timeout = not ready yet!
					return 0;
				}
				//printf("rgb code %d...\n",i);
				usb_cancel_async(rgb_xfers[mCurrentRGBContext]);
			}
			else
			{
				//	printf(",");
			};
		};
		ZeroMemory(&rgb_bufs[mCurrentRGBContext][0], RGB_XFER_SIZE*2);
		int ret = usb_submit_async(rgb_xfers[mCurrentRGBContext], (char*)&rgb_bufs[mCurrentRGBContext][0],  RGB_XFER_SIZE);
		if( ret < 0 )
		{
			printf("error: %s\n", usb_strerror());
			usb_cancel_async(rgb_xfers[mCurrentRGBContext]);
		}
		mCurrentRGBContext = (mCurrentRGBContext + 1) % RGB_NUM_XFERS;
		return retval;

	}

	struct cam_hdr {
		uint8_t magic[2];
		uint16_t len;
		uint16_t cmd;
		uint16_t tag;
	};

	void KinectInternalData::send_init()
	{

		int i, j, ret;
		uint8_t obuf[0x2000];
		uint8_t ibuf[0x2000];
		ZeroMemory(obuf, 0x2000);
		ZeroMemory(ibuf, 0x2000);
		
		cam_hdr *chdr = (cam_hdr *)obuf;
		cam_hdr *rhdr = (cam_hdr *)ibuf;		
		ret = 0;	
		
		ret = usb_control_msg(mDeviceHandle, 0x80, 0x06, 0x3ee, 0, (char*)ibuf, 0x12, 500);
		if (ret <0)
		{
			//	this call is expected to stall!
		};


		chdr->magic[0] = 0x47;
		chdr->magic[1] = 0x4d;

//	Addition by maa nov 16th 2010
//	This table keeps track of which init codes need extra sleep time
		CONST int bs = 1;
		int sleep[num_inits*2] =
		{
			0,0,	//1
			0,0,
			0,0,
			0,0,
			0,bs,	//5
			0,0,	//6
			0,0,
			0,0,
			0,0,
			0,0,
			0,0,	//11
			0,0,
			0,0,
			0,0,
			0,0,	
			0,bs,	//16
			0,0,
			0,0,
			0,0,
			0,0,
			0,bs,	//21
			0,0,
			0,0,
			0,0,
			0,0,
			0,0,	//26
			0,bs,	//27
			0,0,
		};

		for (i=0; i<num_inits; i++) 
		{
			if( sleep[2*i]!=0 )
			{
				Sleep(sleep[2*i]);	//maa
			};

			//Sleep(100);
			//printf("doing init %d\n", i);
			const struct caminit *ip = &inits[i];
			chdr->cmd = ip->command;
			chdr->tag = ip->tag;
			chdr->len = ip->cmdlen / 2;
			memcpy(obuf+sizeof(cam_hdr), ip->cmddata, ip->cmdlen);
			ret = usb_control_msg(mDeviceHandle, 0x40, 0, 0, 0, (char*)obuf, ip->cmdlen + sizeof(cam_hdr), 1600);
			if (ret <0)
			{
				printf("error: %s\n", usb_strerror());
				//return;
			}
			printf("sending init %d from %d... ", i+1, num_inits);
			
			do 
			{
				if( sleep[2*i+1]!=0 )
				{
					Sleep(sleep[2*i+1]);	//maa
				}

				ret = usb_control_msg(mDeviceHandle, 0xc0, 0, 0, 0, (char*)ibuf, 0x200, 1600);
				if (ret<0)
				{
					printf("error: %s\n", usb_strerror());				
				}

			} while (ret == 0);

			if (rhdr->magic[0] != 0x52 || rhdr->magic[1] != 0x42) 
			{
				printf("Bad magic %02x %02x\n", rhdr->magic[0], rhdr->magic[1]);
				continue;
			}
			printf("succes!\n");


			if (rhdr->cmd != chdr->cmd) 
			{
				printf("Bad cmd %02x != %02x\n", rhdr->cmd, chdr->cmd);
				continue;
			}

			if (rhdr->tag != chdr->tag) 
			{
				printf("Bad tag %04x != %04x\n", rhdr->tag, chdr->tag);
				continue;
			}

			if (rhdr->len != (ret-sizeof(*rhdr))/2) 
			{
				printf("Bad len %04x != %04x\n", rhdr->len, (int)(ret-sizeof(*rhdr))/2);
				continue;
			}
			
			if (rhdr->len != (ip->replylen/2) || memcmp(ibuf+sizeof(*rhdr), ip->replydata, ip->replylen)) 
			{
				printf("Expected: ");
				for (j=0; j<ip->replylen; j++) {
					printf("%02x ", ip->replydata[j]);
				}
				printf("\nGot:      ");
				for (j=0; j<(rhdr->len*2); j++) {
					printf("%02x ", ibuf[j+sizeof(*rhdr)]);
				}
				printf("\n");
			}
		}
	}

	void KinectInternalData::cams_init()
	{		
		send_init();
		RunThread();
	}

	KinectInternalData::~KinectInternalData()
	{
		if (mDeviceHandle)
		{
			if (mDeviceHandle_Motor)
			{
				usb_close(mDeviceHandle_Motor);
				mDeviceHandle_Motor = NULL;
			};
			if (mDeviceHandle_Audio)
			{
				usb_close(mDeviceHandle_Audio);
				mDeviceHandle_Motor = NULL;
			};
			Running = false;
			LockDepthThread();
			UnlockDepthThread();
			LockRGBThread();
			UnlockRGBThread();
			usb_reset(mDeviceHandle);
			mParent->KinectDisconnected();
			usb_close(mDeviceHandle);
		};
	};



	void KinectInternalData::RunThread()
	{
		DWORD did, rid ;
		if (DODEPTH)
		{
			HANDLE depththread = CreateThread(NULL,0,DepthThread,this,0,&did);   
			SetThreadPriority(depththread, THREAD_PRIORITY_TIME_CRITICAL);
		};
		if (DORGB)
		{
			HANDLE rgbthread = CreateThread(NULL,0,RGBThread,this,0,&rid);   
			SetThreadPriority(rgbthread, THREAD_PRIORITY_TIME_CRITICAL);
		};
		//		return;

		
		ThreadDone = true;
	};


	void KinectInternalData::OpenDevice(usb_device_t *dev, usb_device_t *motordev)
	{
		mDeviceHandle = usb_open(dev);
		if (!mDeviceHandle) 
		{				
			return;
		}

		//mDeviceHandle_Motor  = NULL;
		mDeviceHandle_Motor = usb_open(motordev); // dont check for null... just dont move when asked and the pointer is null
		
		int ret;
		ret = usb_set_configuration(mDeviceHandle, 1);
		if (ret<0)
		{
			printf("usb_set_configuration error: %s\n", usb_strerror());
			//return;
		}

		ret = usb_claim_interface(mDeviceHandle, 0);
		ret = usb_set_configuration(mDeviceHandle, 1);

		if (ret<0)
		{
			printf("usb_claim_interface error: %s\n", usb_strerror());
			return;
		}

		usb_clear_halt(mDeviceHandle, 0x81);usb_clear_halt(mDeviceHandle, 0x82);
		cams_init();
	};

	void KinectInternalData::SetMotorPosition(double newpos)
	{
		if (mDeviceHandle_Motor)
		{
			if (newpos>1) newpos = 1;if(newpos<0) newpos = 0;
			unsigned char tiltValue = (unsigned char)(newpos*255);
			unsigned short value = (unsigned short)(0xffd0 + tiltValue / 5);
			 //UsbSetupPacket setup = new UsbSetupPacket(0x40, 0x31, mappedValue, 0x0, 0x0);
            //int len = 0;
            //MyUsbDevice.ControlTransfer(ref setup, IntPtr.Zero, 0, out len);
//		cam_hdr *chdr = (cam_hdr *)obuf;
//		cam_hdr *rhdr = (cam_hdr *)ibuf;		
//		ret = 0;	
		
			usb_control_msg(mDeviceHandle_Motor, 0x40, 0x31, value, 0, NULL, 0, 160);
		
		};
	};

	void KinectInternalData::SetLedMode(unsigned short NewMode)
	{
		if (mDeviceHandle_Motor)
		{			
            usb_control_msg(mDeviceHandle_Motor, 0x40, 0x06, NewMode, 0, NULL, 0, 160);
		};
	};

	bool KinectInternalData::GetAcceleroData(float *x, float *y, float *z)
	{
		if (mDeviceHandle_Motor)
		{
			unsigned char outbuf[10];
			if (usb_control_msg(mDeviceHandle_Motor, 0xC0, 0x32, 0, 0, (char*)outbuf, 10, 1000)>0)
			{
				short ix = *(short*)(&outbuf[2]);
				short iy = *(short*)(&outbuf[4]);
				short iz = *(short*)(&outbuf[6]);
				*x = ix/512.0f;			
				*y = iy/512.0f;
				*z = iz/512.0f;
				return true;
			};
		};
		return false;

	};

	KinectInternalData::KinectInternalData(Kinect *inParent)
	{
		mParent = inParent;
		depth_pos = 0;
		rgb_pos = 0;
		mDeviceHandle = NULL;
		mDeviceHandle_Audio = NULL;
		mDeviceHandle_Motor = NULL;
		mCurrentDepthContext = 0;
		mCurrentRGBContext = 0;
		mErrorCount = 0;

		ThreadDone = false;
		Running = true;
//			depth_leftoverbytes= 0;

		DepthPacketCount = 0;
		RGBPacketCount = 0;


		depth_sourcebuf = new uint8_t[1000*1000*3];
		depth_sourcebuf2 = new uint8_t[1000*1000*3];
		depth_frame = new uint16_t[1000*1000*3];
		depth_frame_color = new uint8_t[1000*1000*3];

		rgb_buf= new uint8_t[1000*1000*3];
		rgb_frame= new uint8_t[1000*1000*3];
		rgb_buf2= new uint8_t[1000*1000*3];

		rgb_frame+=1000;
		//rgb_frame2+=1000;

		InitializeCriticalSection(&depth_lock);
		InitializeCriticalSection(&rgb_lock);
		InitializeCriticalSection(&depththread_lock);
		InitializeCriticalSection(&rgbthread_lock);
	}
};