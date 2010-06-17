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
#include "Chrono.h"
#include <Ogre/RenderSystems/Direct3D9/OgreD3D9Texture.h>
#include <Ogre/RenderSystems/Direct3D9/OgreD3D9RenderSystem.h>
#include <Ogre/RenderSystems/Direct3D9/OgreD3D9DeviceManager.h>
#include <Ogre/RenderSystems/Direct3D9/OgreD3D9Device.h>
#include <Ogre/OgrePanelOverlayElement.h>
#include <sstream>

#include <cuda_runtime.h>
#include <cuda_d3d9_interop.h>
#include "GPUSurf.cuh"

using namespace Ogre;
using namespace std;

GPUSurf::GPUSurf(int _nbOctave)
{
	mNbOctave   = _nbOctave;	
	mThreshold  = 0.03f;//15;
	mCamera     = NULL;
	mSceneMgr   = NULL;
	mGrayBuffer = NULL;

	//cuda
	mDeviceFeatureFound = NULL;
	mNbFeatureMax = 4096;

	mThresholdListener = new GPUSurfThresholdListener(this);
	mNbFeatureListener = new GPUSurfNbFeatureListener(this);
}

GPUSurf::~GPUSurf()
{
	exitCuda();
	
	//free texture after
	delete mThresholdListener;
	delete mNbFeatureListener;
}

void GPUSurf::init(int _width, int _height)
{
	mSceneMgr = Ogre::Root::getSingletonPtr()->createSceneManager(ST_GENERIC, "GPGPU/SceneManager");
	mCamera   = mSceneMgr->createCamera("GPGPU/camera");

	mVideoWidth  = _width;
	mVideoHeight = _height;

	int width    = mVideoWidth; 
	int height   = mVideoHeight;
	int hwidth   = (int)width/2;
	int hheight  = (int)height/2;
	int nbOctave = mNbOctave; 

	mGrayTexture = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gray", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, width,  height,  nbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mGxTexture   = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gx",   Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, width,  height,  nbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mGyTexture   = Ogre::TextureManager::getSingleton().createManual("GPGPU/Gy",   Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, width,  height,  nbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mHTexture    = Ogre::TextureManager::getSingleton().createManual("GPGPU/H",    Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, width,  height,  nbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mNMSTexture  = Ogre::TextureManager::getSingleton().createManual("GPGPU/NMS",  Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, hwidth, hheight, nbOctave-1, Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);
	mFeatureCudaTexture = Ogre::TextureManager::getSingleton().createManual("GPGPU/Feature/Cuda", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_1D, mNbFeatureMax, 1, 0, Ogre::PF_FLOAT32_RGBA, Ogre::TU_RENDERTARGET);
	mFeatureTexture     = Ogre::TextureManager::getSingleton().createManual("GPGPU/Feature",      Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_1D, mNbFeatureMax, 1, 0, Ogre::PF_FLOAT32_RGBA, Ogre::TU_RENDERTARGET);	
		
	for (int octaveIndex=0; octaveIndex<nbOctave; ++octaveIndex)
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
	for (int octaveIndex=0; octaveIndex<nbOctave; ++octaveIndex)
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
	for (int octaveIndex=0; octaveIndex<nbOctave; ++octaveIndex)
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
	for (int octaveIndex=0; octaveIndex<nbOctave; ++octaveIndex)
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
	for (int octaveIndex=0; octaveIndex<nbOctave; ++octaveIndex)
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
/*
	//Feature buffer image [4096]
	mFeatureViewport = mFeatureTexture->getBuffer(0, 0)->getRenderTarget()->addViewport(mCamera);
	mFeatureViewport->setOverlaysEnabled(false);
	mFeatureCompositor = Ogre::CompositorManager::getSingleton().addCompositor(mFeatureViewport, "GPGPU/Feature");
	mFeatureCompositor->setEnabled(true);
	mFeatureCompositor->addListener(mNbFeatureListener); //listener for updating nbFeatureFound
*/
	initCuda(_width, _height); //need to be called after mNMS texture creation
}

void GPUSurf::update(const PixelBox &frame)
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

	cudaThreadSynchronize();
	IDirect3DBaseTexture9* texture = static_cast<D3D9TexturePtr>(mNMSTexture)->getTexture();
	mapTextureCuda(texture);

	for (int i=0; i<mNbOctave; ++i)
	{
		mNbFeatureFound += extractFeatureLocationCuda(texture, 
													  mDeviceScanPlan[i],
													  i,
													  mDeviceFeatureCounterPass1[i], 
												      mDeviceFeatureCounterPass2[i], 
#ifdef GPUSURF_HOST_DEBUG
											          mHostFeatureCounterPass1[i], 
											          mHostFeatureCounterPass2[i], 
#endif
											          mDeviceFeatureFound, 
											          mHostFeatureFound, 
											          mNbFeatureFound);		
	}

	unmapTextureCuda(texture);
	cudaThreadSynchronize();

	if (mNbFeatureFound < mNbFeatureMax)
	{		
		cudaMemcpy(mHostFeatureFound, mDeviceFeatureFound, mNbFeatureFound*sizeof(Feature), cudaMemcpyDeviceToHost);
		//exportFeatureBuffer(mHostFeatureFound, "features_before.txt", false);
		
		{
			/*
			IDirect3DBaseTexture9* texture = static_cast<D3D9TexturePtr>(mFeatureCudaTexture)->getTexture();
			mapTextureCuda(texture);

			copyCuda2Tex1D(texture, mDeviceFeatureFound, mNbFeatureFound);

			unmapTextureCuda(texture);
			cudaThreadSynchronize();
			*/
		}
/*
		mFeatureViewport->getTarget()->update();

		{
			IDirect3DBaseTexture9* texture = static_cast<D3D9TexturePtr>(mFeatureTexture)->getTexture();
			mapTextureCuda(texture);

			copyTex1D2Cuda(mDeviceFeatureFound, texture, mNbFeatureFound);

			unmapTextureCuda(texture);
			cudaThreadSynchronize();
		}
		
		cudaMemcpy(mHostFeatureFound, mDeviceFeatureFound, mNbFeatureFound*sizeof(Feature), cudaMemcpyDeviceToHost);
		//exportFeatureBuffer(mHostFeatureFound, "features_after.txt", true);
*/
		unsigned int elapsed = chrono.getTimeElapsed();
		cout << elapsed << "ms (";
		if (elapsed == 0)
			cout << "INF";
		else
			cout << (1000.0f / elapsed*1.0f);
		cout<<"fps) - " <<mNbFeatureFound<< " features"<<endl;

		//cudaMemcpy(mDeviceFeatureFound, mHostFeatureFound, mNbFeatureFound*sizeof(Feature), cudaMemcpyHostToDevice);
	}
	else
	{
		//Too many feature found -> you must change the threshold value
	}
}

void GPUSurf::initCuda(int _width, int _height)
{
	//Device Initialisation
	
	//LPDIRECT3DDEVICE9 device = static_cast<Ogre::D3D9RenderSystem*>(Root::getSingleton().getRenderSystem())->getDevice(); //Ogre 1.6
	LPDIRECT3DDEVICE9 device = static_cast<Ogre::D3D9RenderSystem*>(Root::getSingleton().getRenderSystem())->getDeviceManager()->getDevice(0)->getD3D9Device();
	cudaD3D9SetDirect3DDevice(device);
	cudaThreadSynchronize();

	//Allocating buffer for 2-pass feature location extraction on GPU
	unsigned int width = _width;
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

	//Registering DX9 Texture for CUDA
	registerTextureCuda(static_cast<D3D9TexturePtr>(mNMSTexture)->getTexture());
	registerTextureCuda(static_cast<D3D9TexturePtr>(mFeatureCudaTexture)->getTexture());
	registerTextureCuda(static_cast<D3D9TexturePtr>(mFeatureTexture)->getTexture());

	//Allocating integral image on CPU + GPU
	cudaMalloc((void**)&mDeviceIntegralImage, mVideoWidth*mVideoHeight*sizeof(float));
	mHostIntegralImage = new float[mVideoWidth*mVideoHeight];

	mHostFeatureDescriptor = new float[mNbFeatureMax*64];
}

void GPUSurf::exitCuda()
{
	//Unregistering DX9 Texture
	unregisterTextureCuda(static_cast<D3D9TexturePtr>(mNMSTexture)->getTexture());
	unregisterTextureCuda(static_cast<D3D9TexturePtr>(mFeatureCudaTexture)->getTexture());
	unregisterTextureCuda(static_cast<D3D9TexturePtr>(mFeatureTexture)->getTexture());

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

	//Desallocating buffer for integral image on CPU + GPU
	cudaFree(mDeviceIntegralImage);
	delete[] mHostIntegralImage;

	cudaThreadExit();
}

void GPUSurf::up()   
{ 	
	mThreshold += 0.005f;
	cout << "Threshold.up() " <<mThreshold<< " "<<mNbFeatureFound<<endl;
}
void GPUSurf::down() 
{ 
	mThreshold -= 0.005f; 
	if (mThreshold < 0)
		mThreshold = 0.0000000001f;
	cout << "Threshold.down() " <<mThreshold<<" "<<mNbFeatureFound<<endl;
}

/******************************************************************
				For Visual Debugging only 
 ******************************************************************/

void GPUSurf::createOverlays()
{
	createMaterials();

	std::vector<string> textures;
	textures.push_back("GPGPU/Gray");
	//textures.push_back("GPGPU/Gx");
	//textures.push_back("GPGPU/Gy");
	//textures.push_back("GPGPU/H");
	textures.push_back("GPGPU/NMS");
	 
	OverlayManager& overlayManager = OverlayManager::getSingleton();
	Ogre::Overlay* overlay = overlayManager.create("GPGPU/Debug/Overlay");

	for (unsigned int i=0; i<textures.size(); ++i)
	{
		string texture = textures[i];

		for (int o=0; o<mNbOctave; ++o)
		{
			int pot = 1 << o;
			int width = mVideoWidth / pot;
			int height = mVideoHeight / pot;
			int left = 0; 
			for (int j=0; j<o; ++j)
			{
				int pot = 1 << j;
				left += mVideoWidth / pot;
			}
			int top = i*mVideoHeight;

			std::stringstream panelName, materialName;
			panelName << texture << "/Panel/" << o;
			materialName << texture << "/" << o;

			Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", panelName.str()));
			panel->setMetricsMode(GMM_PIXELS);
			panel->setMaterialName(materialName.str());
			panel->setDimensions((Ogre::Real)width, (Ogre::Real)height);
			panel->setPosition((Ogre::Real)left, (Ogre::Real)top);
			overlay->add2D(panel);
		}
	}
	overlay->show();
}

void GPUSurf::createMaterials()
{
	std::vector<string> textures;
	textures.push_back("GPGPU/Gray");
	textures.push_back("GPGPU/Gx");
	textures.push_back("GPGPU/Gy");
	textures.push_back("GPGPU/H");
	textures.push_back("GPGPU/NMS");

	for (int o=1; o<mNbOctave; ++o)
	{
		std::stringstream materialName;
		materialName << "GPGPU/DownSampling/Mipmap" << o;
		MaterialPtr material = static_cast<MaterialPtr>(MaterialManager::getSingleton().getByName(materialName.str()));
		material.get()->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("octave", (float)o);
	}

	for (unsigned int i=0; i<textures.size(); ++i)
	{
		string texture = textures[i];

		for (int o=0; o<mNbOctave; ++o)
		{
			std::stringstream materialName;
			materialName << texture << "/" << o;

			MaterialPtr material = MaterialManager::getSingleton().create(materialName.str(), ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
			Ogre::Technique *technique = material->createTechnique();
			technique->createPass();
			material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
			material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
			material->getTechnique(0)->getPass(0)->createTextureUnitState(texture);
			material->getTechnique(0)->getPass(0)->setVertexProgram("GPGPU_fixed_vp");
			material->getTechnique(0)->getPass(0)->setFragmentProgram("GPGPU_octave_fp");
			material->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("octave", (float)o);	
			if (texture == "GPGPU/NMS" || texture == "GPGPU/H")
				material->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("scale", Ogre::Vector4(255.0f, 255.0f, 255.0f, 1.0f));
		}
	}
}

GPUSurfThresholdListener::GPUSurfThresholdListener(GPUSurf* _surf)
{
	mSurf = _surf;
}

void GPUSurfThresholdListener::notifyMaterialRender(uint32 pass_id, Ogre::MaterialPtr &mat)
{
	Ogre::Real threshold = mSurf->mThreshold;
	mat->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("threshold", (Ogre::Real)threshold);
}

GPUSurfNbFeatureListener::GPUSurfNbFeatureListener(GPUSurf* _surf)
{
	mSurf = _surf;
}
void GPUSurfNbFeatureListener::notifyMaterialRender(uint32 pass_id, Ogre::MaterialPtr &mat)
{
	Ogre::Real nbFeature = (float)mSurf->mNbFeatureFound;
	mat->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("nbFeature", (Ogre::Real)nbFeature);
}

void GPUSurf::exportFeatureBuffer(Feature* features, const std::string& filename, bool filtered)
{
	ofstream output;
	output.open(filename.c_str());
	if (output.is_open())
	{
		for (unsigned int i=0; i<mNbFeatureFound; ++i)
		{
			if (!filtered || (filtered && features[i].octave > 200))
				output << "["<<i<<"] Feature("<<features[i].x<<", "<<features[i].y<<", "<<features[i].scale<<", "<<features[i].octave<<")"<<endl;
		}
	}
	output.close();
}

unsigned int GPUSurf::getNbFeatureFound()
{
	return mNbFeatureFound;
}

Feature* GPUSurf::getFeatures()
{
	return mHostFeatureFound;
}
