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

#include <Ogre/Ogre.h>
#include "GPUSurf.h"
#include "Feature.h"

#include <cudpp.h>

class GPUSurfApplication;
class GPUSurfThresholdListener;
class GPUSurfNbFeatureListener;

class GPUSurf
{
	friend class GPUSurfThresholdListener;
	friend class GPUSurfNbFeatureListener;

	public:
		GPUSurf(int _nbOctave);
		virtual ~GPUSurf();

		void init(int _width, int _height);
		void createOverlays();
		unsigned int getNbFeatureFound();
		Feature* getFeatures();
		
		void up();
		void down();

		void update(const Ogre::PixelBox &frame);

	protected:
		void createMaterials();

		void exportFeatureBuffer(Feature* features, const std::string& filename, bool filtered);

		std::vector<std::pair<Ogre::Vector2, std::vector<float>>> mReferenceDescriptors;

		int mNbOctave;

		int mVideoWidth;
		int mVideoHeight;		

		unsigned char*     mGrayBuffer; //for Fast Corner Detection

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

		Ogre::Real mThreshold;
		unsigned int mNbFeatureFound; //must be < 4096

		//Cuda [Host = CPU, Device = GPU]
		void initCuda(int _width, int _height);
		void exitCuda();
		unsigned int mNbFeatureMax;
		std::vector<CUDPPHandle> mDeviceScanPlan;
		std::vector<float*> mDeviceFeatureCounterPass1; //size 1 x width
		std::vector<float*> mDeviceFeatureCounterPass2; //size 1 x width
		Feature* mDeviceFeatureFound;                   //size nbFeatures x sizeof(Feature) [in practice nbFeature=4096 to avoid reallocation on GPU)

#ifdef GPUSURF_HOST_DEBUG
		std::vector<float*> mHostFeatureCounterPass1;
		std::vector<float*> mHostFeatureCounterPass2;		
#endif
		Feature* mHostFeatureFound;
		
		GPUSurfThresholdListener* mThresholdListener;
		GPUSurfNbFeatureListener* mNbFeatureListener;

		float* mDeviceIntegralImage;
		float* mHostIntegralImage;
		float* mHostFeatureDescriptor;
		//SurfFeature* mDeviceSurfFeature;
};


class GPUSurfThresholdListener : public Ogre::CompositorInstance::Listener
{
	public:
		GPUSurfThresholdListener(GPUSurf* _surf);
		virtual void notifyMaterialRender(Ogre::uint32 pass_id, Ogre::MaterialPtr &mat);

	protected:
		GPUSurf* mSurf;
};

class GPUSurfNbFeatureListener : public Ogre::CompositorInstance::Listener
{
	public:
		GPUSurfNbFeatureListener(GPUSurf* _surf);
		virtual void notifyMaterialRender(Ogre::uint32 pass_id, Ogre::MaterialPtr &mat);

	protected:
		GPUSurf* mSurf;
};