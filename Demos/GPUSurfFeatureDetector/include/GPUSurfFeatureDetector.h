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

#include "Feature.h"
#include <OgreGPGPU.h>
#include <OgrePixelFormat.h>
#include <OgreTexture.h>
#include <OgreCuda.h>
#include <cudpp.h>

class GPUSurfFeatureDetector
{
	public:
		GPUSurfFeatureDetector(Ogre::Cuda::Root* root, int nbOctave, int nbFeatureMax = 4096);
		~GPUSurfFeatureDetector();
		
		void setThreshold(float threshold);

		void update(Ogre::PixelBox& frame);
		
		unsigned int getNbFeatureFound();
		Feature* getFeatures();

	protected:
		bool resize(int width, int height);
		bool resize(int nbFeature);
		void allocOgreResource();
		void allocCudaResource();
		void freeOgreResource();
		void freeCudaResource();

		int mWidth;
		int mHeight;
		int mNbFeatureMax;
		float mThreshold;
		int mNbOctave;

		int mNbFeatureFound;
		Ogre::GPGPU::Root* mGPGPURoot;

		bool mOgreIsAllocated;
		bool mCudaIsAllocated;
		Ogre::TexturePtr mWebcamTexture;

		//Ogre::Resource

		Ogre::TexturePtr   mGrayTexture;
		Ogre::TexturePtr   mGxTexture;
		Ogre::TexturePtr   mGyTexture;
		Ogre::TexturePtr   mHTexture;
		Ogre::TexturePtr   mNMSTexture;

		//Ogre::GPGPU::Resource
		
		std::vector<Ogre::GPGPU::Result*> mGrayResults;
		std::vector<Ogre::GPGPU::Result*> mGxResults;
		std::vector<Ogre::GPGPU::Result*> mGyResults;
		std::vector<Ogre::GPGPU::Result*> mHResults;
		std::vector<Ogre::GPGPU::Result*> mNMSResults;

		std::vector<Ogre::GPGPU::Operation*> mGrayOperations;
		std::vector<Ogre::GPGPU::Operation*> mGxOperations;
		std::vector<Ogre::GPGPU::Operation*> mGyOperations;
		std::vector<Ogre::GPGPU::Operation*> mHOperations;
		std::vector<Ogre::GPGPU::Operation*> mNMSOperations;

		//Ogre::Cuda::Resource

		Ogre::Cuda::Root*    mCudaRoot;
		Ogre::Cuda::Texture* mCudaNMSTexture;
		Ogre::Cuda::Texture* mCudaFeatureCudaTexture;
		Ogre::Cuda::Texture* mCudaFeatureTexture;

		std::vector<CUDPPHandle> mDeviceScanPlan;
		std::vector<float*> mDeviceFeatureCounterPass1; //size 1 x width
		std::vector<float*> mDeviceFeatureCounterPass2; //size 1 x width
		Feature* mDeviceFeatureFound;                   //size nbFeatures x sizeof(Feature) [in practice nbFeature=4096 to avoid reallocation on GPU)
		Feature* mHostFeatureFound;
};

