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

#include <OgreCuda.h>

#include <OgrePixelFormat.h>
#include <OgreCompositorInstance.h>

#include <cudpp.h>
#include "Feature.h"

namespace GPUSurf
{
	class Plan
	{
		friend class ThresholdListener;
		friend class NbFeatureListener;

		public:
			Plan(Ogre::Cuda::Root* cudaRoot, int width, int height, int nbOctave = 5, int nbFeatureMax = 4096);
			~Plan();

			void update(Ogre::PixelBox& frame);
			void createDebugOverlays();	
			unsigned int getNbFeatureFound();
			Feature* getFeatures();

		protected:
			void allocateOgreResource();
			void allocateCudaResource();
			void freeOgreResource();
			void freeCudaResource();

			void createDebugMaterials();		
			void exportFeatureBuffer(const std::string& filename, Feature* features, unsigned int nbFeature); //To be removed

			int mWidth;
			int mHeight;
			int mNbOctave;
			int mNbFeatureMax;
			float mThreshold;
			int mNbFeatureFound;

			Ogre::TexturePtr   mGrayTexture;
			Ogre::TexturePtr   mGxTexture;
			Ogre::TexturePtr   mGyTexture;
			Ogre::TexturePtr   mHTexture;
			Ogre::TexturePtr   mNMSTexture;
			Ogre::TexturePtr   mFeatureCudaTexture;
			Ogre::TexturePtr   mFeatureTexture;	

			std::vector<Ogre::Viewport*> mGrayViewports;
			std::vector<Ogre::Viewport*> mGxViewports;
			std::vector<Ogre::Viewport*> mGyViewports;
			std::vector<Ogre::Viewport*> mHViewports;
			std::vector<Ogre::Viewport*> mNMSViewports;
			Ogre::Viewport* mFeatureViewport;

			std::vector<Ogre::CompositorInstance*> mGrayCompositors;
			std::vector<Ogre::CompositorInstance*> mGxCompositors;
			std::vector<Ogre::CompositorInstance*> mGyCompositors;
			std::vector<Ogre::CompositorInstance*> mHCompositors;
			std::vector<Ogre::CompositorInstance*> mNMSCompositors;
			Ogre::CompositorInstance* mFeatureCompositor;

			Ogre::Camera*       mCamera;
			Ogre::SceneManager* mSceneMgr;

			ThresholdListener* mThresholdListener;
			NbFeatureListener* mNbFeatureListener;

			Ogre::Cuda::Root*    mCudaRoot;
			Ogre::Cuda::Texture* mCudaNMSTexture;
			Ogre::Cuda::Texture* mCudaFeatureCudaTexture;
			Ogre::Cuda::Texture* mCudaFeatureTexture;

			std::vector<CUDPPHandle> mDeviceScanPlan;
			std::vector<float*> mDeviceFeatureCounterPass1; //size 1 x width
			std::vector<float*> mDeviceFeatureCounterPass2; //size 1 x width
			Feature* mDeviceFeatureFound;                   //size nbFeatures x sizeof(Feature) [in practice nbFeature=4096 to avoid reallocation on GPU)

#ifdef GPUSURF_HOST_DEBUG
			std::vector<float*> mHostFeatureCounterPass1;
			std::vector<float*> mHostFeatureCounterPass2;		
#endif
			Feature* mHostFeatureFound;
	};

	class ThresholdListener : public Ogre::CompositorInstance::Listener
	{
		public:
			ThresholdListener(Plan* plan);
			virtual void notifyMaterialRender(Ogre::uint32 pass_id, Ogre::MaterialPtr &mat);

		protected:
			Plan* mPlan;
	};

	class NbFeatureListener : public Ogre::CompositorInstance::Listener
	{
		public:
			NbFeatureListener(Plan* plan);
			virtual void notifyMaterialRender(Ogre::uint32 pass_id, Ogre::MaterialPtr &mat);

		protected:
			Plan* mPlan;
	};
}