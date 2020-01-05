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

#include "GPUSurfFeatureDetector.h"

#include <OgreTextureManager.h>
#include <OgreMaterialManager.h>
#include <OgreTechnique.h>

#include "GPUSurf.cuh"

#include <gl/GL.h>

GPUSurfFeatureDetector::GPUSurfFeatureDetector(Ogre::Cuda::Root* root, int nbOctave, int nbFeatureMax)
: mCudaRoot(root)
{
	mOgreIsAllocated = false;
	mCudaIsAllocated = false;

	mWidth          = 0;
	mHeight         = 0;
	mNbFeatureMax   = nbFeatureMax;
	mNbOctave       = nbOctave;
	mThreshold      = 0.003f;

	mNbFeatureFound = 0;
	mGPGPURoot = new Ogre::GPGPU::Root;

	mWebcamTexture = Ogre::TextureManager::getSingleton().createManual("WebcamVideoTexture", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, 640,  480,  0, Ogre::PF_R8G8B8, Ogre::TU_RENDERTARGET);

	//Create Webcam Material
	Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().create("WebcamVideoMaterial", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::Technique *technique = material->createTechnique();
	technique->createPass();
	material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
	material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
	material->getTechnique(0)->getPass(0)->createTextureUnitState("WebcamVideoTexture");
}

GPUSurfFeatureDetector::~GPUSurfFeatureDetector()
{
	delete mGPGPURoot;
}

void GPUSurfFeatureDetector::setThreshold(float threshold)
{
	mThreshold = threshold;
}

void GPUSurfFeatureDetector::update(Ogre::PixelBox& frame)
{
	mNbFeatureFound = 0;

	int width = frame.getWidth();
	int height = frame.getHeight();
	if (resize(frame.getWidth(), frame.getHeight()))
	{
		std::cout << "Warning resizing to " << width << "x" <<height << std::endl;
	}

	mWebcamTexture->getBuffer(0, 0)->blitFromMemory(frame);

	for (int i=0; i<mNbOctave; ++i)
	{
		mGPGPURoot->compute(mGrayResults[i], mGrayOperations[i]);
		mGPGPURoot->compute(mGxResults[i],   mGxOperations[i]);
		mGPGPURoot->compute(mGyResults[i],   mGyOperations[i]);
		mGPGPURoot->compute(mHResults[i],    mHOperations[i]);
		mNMSOperations[i]->setParameter("threshold", (float) mThreshold);
		mGPGPURoot->compute(mNMSResults[i],  mNMSOperations[i]);
	}
	/*
	for (int i=0; i<mNbOctave; ++i)
	{
		std::stringstream name;
		name << "gaussian_" << i<< ".png";
		mGrayResults[i]->save(name.str());
	}
	*/
	glFinish();

	mCudaRoot->synchronize();
	//std::cout << "0 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;

	for (int i=0; i<mNbOctave; ++i)
	{
		mCudaNMSTexture->map();
		mCudaRoot->synchronize();

		//std::cout << "1 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;
		
		Ogre::Cuda::TextureDeviceHandle textureHandle = mCudaNMSTexture->getDeviceHandle(0, i);
		mCudaRoot->synchronize();

		//std::cout << "2 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;
		mCudaNMSTexture->updateReading(textureHandle);
		mCudaRoot->synchronize();

		//std::cout << "3 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;
		mNbFeatureFound += extractFeatureLocationCuda(textureHandle.width, textureHandle.height, textureHandle.getPointer(), 		
			mDeviceScanPlan[i],
			i,
			mDeviceFeatureCounterPass1[i], 
			mDeviceFeatureCounterPass2[i], 
			mDeviceFeatureFound,
			mNbFeatureFound);

		mCudaRoot->synchronize();
		mCudaNMSTexture->unmap();

		//std::cout << "6 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;

		//std::cout << "4 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;
		mCudaRoot->synchronize();
		//std::cout << "5 Cuda Error : " << mCudaRoot->getLastError() <<std::endl;
	}	

	mCudaRoot->synchronize();
	if (mNbFeatureFound < mNbFeatureMax)
	{		
		cudaMemcpy(mHostFeatureFound, mDeviceFeatureFound, mNbFeatureFound*sizeof(Feature), cudaMemcpyDeviceToHost);
	}
	else
	{
		mNbFeatureFound = 0;
		//should launch again with a bigger feature buffer...
	}
}

bool GPUSurfFeatureDetector::resize(int width, int height)
{
	if (mWidth < width || mHeight < height)		 
	{
		mWidth  = std::max(mWidth,  width);
		mHeight = std::max(mHeight, height);
		freeCudaResource();
		freeOgreResource();		
		allocOgreResource();
		allocCudaResource();

		return true;
	}
	else
		return false;
}

void GPUSurfFeatureDetector::allocOgreResource()
{
	int hwidth  = (int)mWidth  / 2;
	int hheight = (int)mHeight / 2;

	//create ogre tex
	mGrayTexture = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gray", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mGxTexture   = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gx",   Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mGyTexture   = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gy",   Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mHTexture    = Ogre::TextureManager::getSingleton().createManual("GPGPU/H",    Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, mWidth,  mHeight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mNMSTexture  = Ogre::TextureManager::getSingleton().createManual("GPGPU/NMS",  Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, hwidth,  hheight,  mNbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	//mFeatureCudaTexture = Ogre::TextureManager::getSingleton().createManual("GPGPU/Feature/Cuda", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_1D, mNbFeatureMax, 1, 0, Ogre::PF_FLOAT32_RGBA, Ogre::TU_RENDERTARGET);
	//mFeatureTexture     = Ogre::TextureManager::getSingleton().createManual("GPGPU/Feature",      Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_1D, mNbFeatureMax, 1, 0, Ogre::PF_FLOAT32_RGBA, Ogre::TU_RENDERTARGET);	

	//create operation
	for (int i=0; i<mNbOctave; ++i)
	{		
		mGrayResults.push_back(mGPGPURoot->createResult(mGrayTexture->getBuffer(0, i)));
		mGxResults.push_back(mGPGPURoot->createResult(mGxTexture->getBuffer(0,     i)));
		mGyResults.push_back(mGPGPURoot->createResult(mGyTexture->getBuffer(0,     i)));
		mHResults.push_back(mGPGPURoot->createResult(mHTexture->getBuffer(0,       i)));
		mNMSResults.push_back(mGPGPURoot->createResult(mNMSTexture->getBuffer(0,   i)));

		std::stringstream materialName;		
		{
			if (i == 0)
				materialName << "GPGPU/RGB2Gray/Mipmap0";
			else 
				materialName << "GPGPU/DownSampling/Mipmap" << i;
			Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().getByName(materialName.str());
			mat->load();
			materialName.str("");
			mGrayOperations.push_back(mGPGPURoot->createOperation(mat->getBestTechnique()->getPass(0)));
		}
		{
			materialName << "GPGPU/GaussianX/Mipmap" << i;
			Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().getByName(materialName.str());
			mat->load();
			materialName.str("");
			mGxOperations.push_back(mGPGPURoot->createOperation(mat->getBestTechnique()->getPass(0)));
		}
		{
			materialName << "GPGPU/GaussianY/Mipmap" << i;
			Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().getByName(materialName.str());
			mat->load();
			materialName.str("");
			mGyOperations.push_back(mGPGPURoot->createOperation(mat->getBestTechnique()->getPass(0)));
		}
		{
			materialName << "GPGPU/Hessian/Mipmap" << i;
			Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().getByName(materialName.str());
			mat->load();
			materialName.str("");
			mHOperations.push_back(mGPGPURoot->createOperation(mat->getBestTechnique()->getPass(0)));
		}
		{
			materialName << "GPGPU/NMS/Mipmap" << i;
			Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().getByName(materialName.str());
			mat->load();
			materialName.str("");
			mNMSOperations.push_back(mGPGPURoot->createOperation(mat->getBestTechnique()->getPass(0)));
		}
	}
	mOgreIsAllocated = true;
}

void GPUSurfFeatureDetector::allocCudaResource()
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

	mCudaIsAllocated = true;
}

void GPUSurfFeatureDetector::freeOgreResource()
{
	if (mOgreIsAllocated)
	{

	}
}

void GPUSurfFeatureDetector::freeCudaResource()
{
	if (mCudaIsAllocated)
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
		}

		//Desallocating buffer for Feature location on GPU + CPU
		cudaFree(mDeviceFeatureFound);
		delete[] mHostFeatureFound;
	}
}

unsigned int GPUSurfFeatureDetector::getNbFeatureFound()
{
	return mNbFeatureFound;
}

Feature* GPUSurfFeatureDetector::getFeatures()
{
	return mHostFeatureFound;
}