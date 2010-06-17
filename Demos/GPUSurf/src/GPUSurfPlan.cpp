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

#include "GPUSurf.h"

#include "OgreCuda.h"

#include <OgreRoot.h>
#include <OgreHardwarePixelBuffer.h>
#include <OgreCompositorManager.h>

#include "Chrono.h"

#include "GPUSurf.cuh"

#include <gl/GL.h>

using namespace GPUSurf;

Plan::Plan(Ogre::Cuda::Root* cudaRoot, int width, int height, int nbOctave, int nbFeatureMax)
{
	mCudaRoot       = cudaRoot;
	mWidth          = width;
	mHeight         = height;
	mNbOctave       = nbOctave;	
	mNbFeatureMax   = nbFeatureMax;
	mNbFeatureFound = 0;
	mThreshold      = 0.003f;

	mDeviceFeatureFound = NULL;	
	mSceneMgr           = NULL;
	mCamera             = NULL;

	//Listener need to be created before OgreResource are allocated
	mThresholdListener = new ThresholdListener(this);
	mNbFeatureListener = new NbFeatureListener(this);

	allocateOgreResource();
	allocateCudaResource();
}

Plan::~Plan()
{
	freeCudaResource();
	freeOgreResource();

	delete mThresholdListener;
	delete mNbFeatureListener;
}

void Plan::update(Ogre::PixelBox& frame)
{
	Chrono chrono;
	chrono.start();
	
	mNbFeatureFound = 0;

	for (int i=0; i<mNbOctave; ++i)
	{
		//RGB -> Gray & Downsampling
		mGrayViewports[i]->getTarget()->update();

		//Gray -> Gx & Gy
		mGxViewports[i]->getTarget()->update();
		mGyViewports[i]->getTarget()->update();
		
		//Gaussian -> det[Hessian]
		mHViewports[i]->getTarget()->update();
		
		//det[Hessian] -> Non Maximum Suppression (NMS)
		mNMSViewports[i]->getTarget()->update();
	}

	mCudaRoot->synchronize();
	mCudaNMSTexture->map();

	for (int i=0; i<mNbOctave; ++i)
	{
		Ogre::Cuda::TextureDeviceHandle textureHandle = mCudaNMSTexture->getDeviceHandle(0, i);
		mCudaNMSTexture->updateReading(textureHandle);
		mNbFeatureFound += extractFeatureLocationCuda(textureHandle.width, textureHandle.height, textureHandle.getPointer(), 		
													  mDeviceScanPlan[i],
													  i,
													  mDeviceFeatureCounterPass1[i], 
												      mDeviceFeatureCounterPass2[i], 
#ifdef GPUSURF_HOST_DEBUG
											          mHostFeatureCounterPass1[i], 
											          mHostFeatureCounterPass2[i], 
#endif
											          mDeviceFeatureFound,
											          mNbFeatureFound);	
	}
	mCudaNMSTexture->unmap();
	mCudaRoot->synchronize();

	if (mNbFeatureFound < mNbFeatureMax)
	{		
		cudaMemcpy(mHostFeatureFound, mDeviceFeatureFound, mNbFeatureFound*sizeof(Feature), cudaMemcpyDeviceToHost);
		//exportFeatureBuffer("features_before.txt", mHostFeatureFound, mNbFeatureFound);
		/*
		{						
			mCudaFeatureCudaTexture->map();
			Ogre::Cuda::TextureDeviceHandle textureHandle = mCudaFeatureCudaTexture->getDeviceHandle(0, 0);
			copyCuda2Tex1D(textureHandle.width, textureHandle.height, textureHandle.getPointer(), mDeviceFeatureFound, mNbFeatureFound);
			mCudaFeatureCudaTexture->updateWriting(textureHandle);
			mCudaFeatureCudaTexture->unmap();
			cudaThreadSynchronize();			
		}

		mFeatureViewport->getTarget()->update();
		
		{
			mCudaFeatureTexture->map();
			Ogre::Cuda::TextureDeviceHandle textureHandle = mCudaFeatureCudaTexture->getDeviceHandle(0, 0);
			copyTex1D2Cuda(mDeviceFeatureFound, mNbFeatureMax, 1, mCudaFeatureTexture, mNbFeatureFound);
			mCudaFeatureTexture->unmap();
			cudaThreadSynchronize();
		}
		
		cudaMemcpy(mHostFeatureFound, mDeviceFeatureFound, mNbFeatureFound*sizeof(Feature), cudaMemcpyDeviceToHost);
		*/
		//exportFeatureBuffer("features_after.txt", mHostFeatureFound, mNbFeatureFound);
	}
	else
	{
		//Too many feature found -> you must change the threshold value
	}


	unsigned int elapsed = chrono.getTimeElapsed();
	std::cout << elapsed << "ms (";
	if (elapsed == 0)
		std::cout << "INF";
	else
		std::cout << (1000.0f / elapsed*1.0f);
	std::cout<<"fps) - " <<mNbFeatureFound<<std::endl;
}

void Plan::allocateOgreResource()
{
	mSceneMgr = Ogre::Root::getSingletonPtr()->createSceneManager(Ogre::ST_GENERIC, "GPGPU/SceneManager");
	mCamera   = mSceneMgr->createCamera("GPGPU/camera");
	mCamera->setUseIdentityProjection(true);
	mCamera->setUseIdentityView(true);
	mCamera->setNearClipDistance(1);

	int hwidth  = mWidth / 2;
	int hheight = mHeight / 2;

	mGrayTexture = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gray", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mGxTexture   = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gx",   Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mGyTexture   = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gy",   Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mHTexture    = Ogre::TextureManager::getSingleton().createManual("GPGPU/H",    Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mNMSTexture  = Ogre::TextureManager::getSingleton().createManual("GPGPU/NMS",  Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, hwidth, hheight, mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mFeatureCudaTexture = Ogre::TextureManager::getSingleton().createManual("GPGPU/Feature/Cuda", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_1D, mNbFeatureMax, 1, 0, Ogre::PF_FLOAT32_RGBA, Ogre::TU_RENDERTARGET);
	mFeatureTexture     = Ogre::TextureManager::getSingleton().createManual("GPGPU/Feature",      Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_1D, mNbFeatureMax, 1, 0, Ogre::PF_FLOAT32_RGBA, Ogre::TU_RENDERTARGET);
	
	for (int octaveIndex=0; octaveIndex<mNbOctave; ++octaveIndex)
	{
		mGrayTexture->getBuffer(0, octaveIndex)->getRenderTarget()->setAutoUpdated(false);
		mGxTexture->getBuffer(0,  octaveIndex)->getRenderTarget()->setAutoUpdated(false);
		mGyTexture->getBuffer(0,  octaveIndex)->getRenderTarget()->setAutoUpdated(false);
		mHTexture->getBuffer(0,   octaveIndex)->getRenderTarget()->setAutoUpdated(false);
		mNMSTexture->getBuffer(0, octaveIndex)->getRenderTarget()->setAutoUpdated(false);
	}
	mFeatureCudaTexture->getBuffer(0, 0)->getRenderTarget()->setAutoUpdated(false);
	mFeatureTexture->getBuffer(0, 0)->getRenderTarget()->setAutoUpdated(false);

	//RGB -> Gray & Downsampling
	for (int octaveIndex=0; octaveIndex<mNbOctave; ++octaveIndex)
	{
		Ogre::Viewport* viewport = mGrayTexture->getBuffer(0, octaveIndex)->getRenderTarget()->addViewport(mCamera);
		viewport->setOverlaysEnabled(false);
		Ogre::CompositorInstance* compositor = NULL;

		if (octaveIndex == 0)
			compositor = Ogre::CompositorManager::getSingleton().addCompositor(viewport, "GPGPU/RGB2Gray/Mipmap0");
		else 
		{
			std::stringstream compositorName;
			compositorName << "GPGPU/DownSampling/Mipmap" << octaveIndex;
			compositor = Ogre::CompositorManager::getSingleton().addCompositor(viewport, compositorName.str());
		}

		compositor->setEnabled(true);
		mGrayViewports.push_back(viewport);
		mGrayCompositors.push_back(compositor);
	}

	//Gray -> GaussianX -> GaussianY (on each mipmap)
	for (int octaveIndex=0; octaveIndex<mNbOctave; ++octaveIndex)
	{
		std::stringstream compositorName;
		compositorName << "GPGPU/GaussianX/Mipmap" << octaveIndex;

		//Gray -> Gx (gaussian-x)
		Ogre::Viewport* viewportGx = mGxTexture->getBuffer(0, octaveIndex)->getRenderTarget()->addViewport(mCamera);
		viewportGx->setOverlaysEnabled(false);
		Ogre::CompositorInstance* compositorGx = Ogre::CompositorManager::getSingleton().addCompositor(viewportGx, compositorName.str());
		compositorGx->setEnabled(true);	

		compositorName.str("");
		compositorName << "GPGPU/GaussianY/Mipmap" << octaveIndex;

		//Gx -> Gy (gaussian-y)
		Ogre::Viewport* viewportGy = mGyTexture->getBuffer(0, octaveIndex)->getRenderTarget()->addViewport(mCamera);
		viewportGy->setOverlaysEnabled(false);
		Ogre::CompositorInstance* compositorGy = Ogre::CompositorManager::getSingleton().addCompositor(viewportGy, compositorName.str());
		compositorGy->setEnabled(true);

		mGxViewports.push_back(viewportGx);
		mGyViewports.push_back(viewportGy);
		mGxCompositors.push_back(compositorGx);
		mGyCompositors.push_back(compositorGy);
	}

	//Gaussian -> det[Hessian] (on each mipmap)
	for (int octaveIndex=0; octaveIndex<mNbOctave; ++octaveIndex)
	{
		std::stringstream compositorName;
		compositorName << "GPGPU/Hessian/Mipmap" << octaveIndex;

		//Gaussian -> Hessian
		Ogre::Viewport* viewport = mHTexture->getBuffer(0, octaveIndex)->getRenderTarget()->addViewport(mCamera);
		viewport->setOverlaysEnabled(false);
		Ogre::CompositorInstance* compositor = Ogre::CompositorManager::getSingleton().addCompositor(viewport, compositorName.str());
		compositor->setEnabled(true);			

		mHViewports.push_back(viewport);
		mHCompositors.push_back(compositor);
	}

	//det[Hessian] -> Non Maximum Suppression (on each mipmap)
	for (int octaveIndex=0; octaveIndex<mNbOctave; ++octaveIndex)
	{
		std::stringstream compositorName;
		compositorName << "GPGPU/NMS/Mipmap" << octaveIndex;

		//Non Maximum Suppression on det[Hessian]
		Ogre::Viewport* viewport = mNMSTexture->getBuffer(0, octaveIndex)->getRenderTarget()->addViewport(mCamera);
		viewport->setOverlaysEnabled(false);
		Ogre::CompositorInstance* compositor = Ogre::CompositorManager::getSingleton().addCompositor(viewport, compositorName.str());
		compositor->setEnabled(true);	
		compositor->addListener(mThresholdListener); //listener for changing threshold value

		mNMSViewports.push_back(viewport);
		mNMSCompositors.push_back(compositor);
	}

	//Feature buffer image [4096]
	mFeatureViewport = mFeatureTexture->getBuffer(0, 0)->getRenderTarget()->addViewport(mCamera);
	mFeatureViewport->setOverlaysEnabled(false);
	mFeatureCompositor = Ogre::CompositorManager::getSingleton().addCompositor(mFeatureViewport, "GPGPU/Feature");
	mFeatureCompositor->setEnabled(true);
	mFeatureCompositor->addListener(mNbFeatureListener); //listener for updating nbFeatureFound
}

void Plan::allocateCudaResource()
{
	//Allocating buffer for 2-pass feature location extraction on GPU
	unsigned int width = mWidth;
	for (int i=0; i<mNbOctave; ++i)
	{		
		width /= 2;
		float* devicePass1 = NULL;
		float* devicePass2 = NULL;

		cudaMalloc((void**)&devicePass1, width*sizeof(float));
		cudaMalloc((void**)&devicePass2, width*sizeof(float));

		CUDPPHandle scanPlan;
		CUDPPConfiguration config = { CUDPP_SCAN, CUDPP_ADD, CUDPP_FLOAT, CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE };
		cudppPlan(&scanPlan, config, width, 1, 0);

		mDeviceFeatureCounterPass1.push_back(devicePass1);
		mDeviceFeatureCounterPass2.push_back(devicePass2);
		mDeviceScanPlan.push_back(scanPlan);

#ifdef GPUSURF_HOST_DEBUG
		float* hostPass1 = new float[width];
		float* hostPass2 = new float[width];
		mHostFeatureCounterPass1.push_back(hostPass1);
		mHostFeatureCounterPass2.push_back(hostPass2);
#endif

	}

	//Allocating buffer for Feature location on GPU + CPU
	cudaMalloc((void**)&mDeviceFeatureFound, mNbFeatureMax*sizeof(Feature));
	mHostFeatureFound = new Feature[mNbFeatureMax];
	
	//Creating Cuda Texture
	mCudaNMSTexture         = mCudaRoot->getTextureManager()->createTexture(mNMSTexture);
	//mCudaFeatureCudaTexture = mCudaRoot->getTextureManager()->createTexture(mFeatureCudaTexture);
	//mCudaFeatureTexture     = mCudaRoot->getTextureManager()->createTexture(mFeatureTexture);

	//Registering Texture for CUDA
	mCudaNMSTexture->registerForCudaUse();
	//mCudaFeatureCudaTexture->registerForCudaUse();
	//mCudaFeatureTexture->registerForCudaUse();
}

void Plan::freeCudaResource()
{	
	//Unregistering Texture
	mCudaNMSTexture->unregister();
	//mCudaFeatureCudaTexture->unregister();
	//mCudaFeatureTexture->unregister();

	//Deleting Texture
	mCudaRoot->getTextureManager()->destroyTexture(mCudaNMSTexture);
	//mCudaRoot->getTextureManager()->destroyTexture(mCudaFeatureCudaTexture);
	//mCudaRoot->getTextureManager()->destroyTexture(mCudaFeatureTexture);

	//Desallocating buffer for 2-pass feature location extraction on GPU
	for (int i=0; i<mNbOctave; ++i)
	{
		cudaFree(mDeviceFeatureCounterPass1[i]);
		cudaFree(mDeviceFeatureCounterPass2[i]);

		cudppDestroyPlan(mDeviceScanPlan[i]);

#ifdef GPUSURF_HOST_DEBUG
		delete[] mHostFeatureCounterPass1[i];
		delete[] mHostFeatureCounterPass2[i];
#endif

	}

	//Desallocating buffer for Feature location on GPU + CPU
	cudaFree(mDeviceFeatureFound);
	delete[] mHostFeatureFound;
}

void Plan::freeOgreResource()
{

}


ThresholdListener::ThresholdListener(Plan* plan)
: mPlan(plan)
{}

void ThresholdListener::notifyMaterialRender(Ogre::uint32 pass_id, Ogre::MaterialPtr &mat)
{
	Ogre::Real threshold = mPlan->mThreshold;
	mat->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("threshold", (Ogre::Real)threshold);
}

NbFeatureListener::NbFeatureListener(Plan* plan)
: mPlan(plan)
{}

void NbFeatureListener::notifyMaterialRender(Ogre::uint32 pass_id, Ogre::MaterialPtr &mat)
{
	int nbFeature = mPlan->mNbFeatureFound;
	mat->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("nbFeature", (Ogre::Real)nbFeature);
}

void Plan::exportFeatureBuffer(const std::string& filename, Feature* features, unsigned int nbFeature)
{
	std::ofstream output;
	output.open(filename.c_str());
	if (output.is_open())
	{
		for (unsigned int i=0; i<nbFeature; ++i)
		{
			output << "[" << i << "] " << features[i].x << " " << features[i].y << std::endl;
		}
	}
	output.close();
}

unsigned int Plan::getNbFeatureFound()
{
	return mNbFeatureFound;
}

Feature* Plan::getFeatures()
{
	return mHostFeatureFound;
}
