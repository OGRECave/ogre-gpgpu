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

#include "OgreCuda.h"
#include "OgreCudaD3D9.h"
#include "OgreCudaD3D10.h"
#include "OgreCudaGL.h"

#include <cuda_runtime.h>

using namespace Ogre::Cuda;

//Root

Root::Root()
: mTextureManager(NULL), mVertexBufferManager(NULL)
{
	mLastCudaError = cudaSuccess;
	mCudaStream    = NULL;
}

Root::~Root()
{
	delete mTextureManager;
	delete mVertexBufferManager;
}

void Root::shutdown()
{
	cudaThreadExit();
}

void Root::synchronize()
{
	cudaThreadSynchronize();
}

TextureManager* Root::getTextureManager()
{
	return mTextureManager;
}

VertexBufferManager* Root::getVertexBufferManager()
{
	return mVertexBufferManager;
}

void Root::map(std::vector<Ogre::Cuda::Ressource*> ressources)
{
	mCudaStream = NULL;
	std::vector<struct cudaGraphicsResource*> cudaRessources;
	for (unsigned int i=0; i<ressources.size(); ++i)
		cudaRessources.push_back((ressources[i])->mCudaRessource);
	cudaGraphicsMapResources(cudaRessources.size(), &(cudaRessources[0]), mCudaStream);
}

void Root::unmap(std::vector<Ogre::Cuda::Ressource*> ressources)
{
	std::vector<struct cudaGraphicsResource*> cudaRessources;
	for (unsigned int i=0; i<ressources.size(); ++i)
		cudaRessources.push_back((ressources[i])->mCudaRessource);
	cudaGraphicsUnmapResources(cudaRessources.size(), &(cudaRessources[0]), mCudaStream);
}

bool Root::isCudaStatusOK()
{
	mLastCudaError = cudaGetLastError();
	return mLastCudaError == cudaSuccess;
}

std::string Root::getErrorMessage()
{
	return std::string(cudaGetErrorString(mLastCudaError));
}

std::string Root::getLastError()
{
	return std::string(cudaGetErrorString(cudaGetLastError()));
}

Root* Root::createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
{
	std::string renderSystemName = renderSystem->getName();

	if (renderSystemName == "OpenGL Rendering Subsystem")
		return new Ogre::Cuda::GLRoot(renderWindow);
	else if (renderSystemName == "Direct3D9 Rendering Subsystem")
		return new Ogre::Cuda::D3D9Root(renderWindow);
	else if (renderSystemName == "Direct3D10 Rendering Subsystem")
		return new Ogre::Cuda::D3D10Root(renderWindow);

	return NULL;
}

void Root::destroyRoot(Root* root)
{
	delete root;
	root = NULL;
}

int Root::getDeviceCount()
{
	int deviceCount = -1;
	cudaGetDeviceCount(&deviceCount);

	return deviceCount;
}

DeviceProperties Root::getDeviceProperties(int index)
{
	int deviceCount = Root::getDeviceCount();
	if (index < deviceCount)
	{
		cudaDeviceProp deviceProp;
		memset(&deviceProp, 0, sizeof(deviceProp));
		if (cudaGetDeviceProperties(&deviceProp, index) == cudaSuccess)
		{
			return DeviceProperties(deviceProp);
		}
	}
	return DeviceProperties();
}

int Root::getCudaRuntimeVersion()
{
	int version = -1;
	cudaRuntimeGetVersion(&version);

	return version;
}

int Root::getVideoDriverVersion()
{
	int version = -1;
	cudaDriverGetVersion(&version);

	return version;
}

DeviceProperties::DeviceProperties()
{
	name               = "UNKNOWN";
	totalGlobalMem     = 0;
	sharedMemPerBlock  = 0;
	regsPerBlock       = 0;
	warpSize           = 0;
	memPitch           = 0;
	maxThreadsPerBlock = 0;
	for (unsigned int i=0; i<3; ++i)
	{
		maxThreadsDim[i] = 0;
		maxGridSize[i]   = 0;
	}
	clockRate                = 0;
	totalConstMem            = 0;
	major                    = -1;
	minor                    = -1;
	textureAlignment         = 0;
	deviceOverlap            = -1;
	multiProcessorCount      = 0;
	kernelExecTimeoutEnabled = 0;
	integrated               = 0;
	canMapHostMemory         = false;
	computeMode              = 0;
}

DeviceProperties::DeviceProperties(const cudaDeviceProp& prop)
{
	name               = std::string(prop.name);
	totalGlobalMem     = prop.totalGlobalMem;
	sharedMemPerBlock  = prop.sharedMemPerBlock;
	regsPerBlock       = prop.regsPerBlock;
	warpSize           = prop.warpSize;
	memPitch           = prop.memPitch;
	maxThreadsPerBlock = prop.maxThreadsPerBlock;
	for (unsigned int i=0; i<3; ++i)
	{
		maxThreadsDim[i] = prop.maxThreadsDim[i];
		maxGridSize[i]   = prop.maxGridSize[i];
	}
	clockRate                = prop.clockRate;
	totalConstMem            = prop.totalConstMem;
	major                    = prop.major;
	minor                    = prop.minor;
	textureAlignment         = prop.textureAlignment;
	deviceOverlap            = prop.deviceOverlap;
	multiProcessorCount      = prop.multiProcessorCount;
	kernelExecTimeoutEnabled = prop.kernelExecTimeoutEnabled;
	integrated               = prop.integrated;
	canMapHostMemory         = prop.canMapHostMemory == 1;
	computeMode              = prop.computeMode;
}

std::ostream& operator <<(std::ostream& output, const DeviceProperties& prop)
{
	if (prop.name != "UNKNOWN")
	{
		output << "Device Name : " << prop.name << std::endl;
		output << "**************************************" << std::endl;
		output << "Total Global Memory : " << (prop.totalGlobalMem/1024) <<" KB" << std::endl;;
		output << "Shared memory available per block : " << (prop.sharedMemPerBlock/1024) << " KB" << std::endl;
		output << "Number of registers per thread block : " << prop.regsPerBlock << std::endl;
		output << "Warp size in threads : " << prop.warpSize << std::endl;
		output << "Memory Pitch : " << prop.memPitch << " bytes" << std::endl;
		output << "Maximum threads per block : " << prop.maxThreadsPerBlock << std::endl;
		output << "Maximum Thread Dimension (block) : " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
		output << "Maximum Thread Dimension (grid) : " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
		output << "Total constant memory : " << prop.totalConstMem << " bytes" << std::endl;
		output << "CUDA version : " << prop.major << "." 
			<< prop.minor <<std::endl;
		output << "Clock rate : " << prop.clockRate << " KHz" << std::endl;
		output << "Texture Alignment : "<< prop.textureAlignment << " bytes" << std::endl;
		output << "Device Overlap : " << (prop.deviceOverlap ? "Allowed" : "Not Allowed") << std::endl;
		output << "Number of Multi processors : " << prop.multiProcessorCount << std::endl;
	}
	return output;
}

//Texture

Texture::Texture(Ogre::TexturePtr texture)
: mTexture(texture)
{}

TextureDeviceHandle Texture::getDeviceHandle(unsigned int face, unsigned int mipmap)
{
	unsigned int index = getIndex(face, mipmap);	
	cudaGraphicsSubResourceGetMappedArray(&mDevicePtrs[index].mCudaArray, mCudaRessource, face, mipmap);

	return mDevicePtrs[index];	
}

void Texture::update(TextureDeviceHandle& mem)
{
	cudaMemcpyToArray(mem.mCudaArray, 0, 0, mem.linearMemory, mem.pitch * mem.height, cudaMemcpyDeviceToDevice);
}

unsigned int Texture::getIndex(unsigned int face, unsigned int mipmap)
{
	return face*(mTexture->getNumMipmaps()+1) + mipmap;
}

Ogre::Vector2 Texture::getDimensions(unsigned int face, unsigned int mipmap)
{
	unsigned int index = getIndex(face, mipmap);
	return Ogre::Vector2((Ogre::Real)mDevicePtrs[index].width, (Ogre::Real)mDevicePtrs[index].height);
}

Ogre::Cuda::RessourceType Texture::getType()
{
	return Ogre::Cuda::TEXTURE_RESSOURCE;
}

void Texture::allocate()
{
	size_t pixelSize = Ogre::PixelUtil::getNumElemBits(mTexture->getFormat()) / 8;

	for (unsigned int i=0; i<mTexture->getNumFaces(); ++i)
	{
		for (unsigned int j=0; j<mTexture->getNumMipmaps()+1; ++j)
		{
			HardwarePixelBufferSharedPtr buffer = mTexture->getBuffer(i, j);
			unsigned int width  = buffer->getWidth();
			unsigned int height = buffer->getHeight();
			
			size_t pitch = 0;
			void* linearMemory = NULL;
			cudaMallocPitch(&linearMemory, &pitch, width * pixelSize, height);
			mDevicePtrs.push_back(Ogre::Cuda::TextureDeviceHandle(width, height, pitch, linearMemory));
		}
	}
}

void Texture::unregister()
{
	Ressource::unregister();
	
	unsigned int index = 0;
	for (unsigned int i=0; i<mTexture->getNumFaces(); ++i)
	{
		for (unsigned int j=0; j<mTexture->getNumMipmaps()+1; ++j)
		{
			cudaFree(mDevicePtrs[index].linearMemory);
			index++;
		}
	}
}

//VertexBuffer

VertexBuffer::VertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: mVertexBuffer(vertexBuffer)
{}

void* VertexBuffer::getPointer()
{
	size_t size = 0;
	void* devicePtr = NULL;
	cudaGraphicsResourceGetMappedPointer((void **)&devicePtr, &size, mCudaRessource);

	return devicePtr;
}

Ogre::Cuda::RessourceType VertexBuffer::getType()
{
	return Ogre::Cuda::VERTEXBUFFER_RESSOURCE;
}

//Ressource

Ressource::Ressource()
{
	mCudaRessource    = NULL;
	mCudaStream       = NULL;
}

void Ressource::unregister()
{
	cudaGraphicsUnregisterResource(mCudaRessource);
}

void Ressource::map()
{
	cudaGraphicsMapResources(1, &mCudaRessource, mCudaStream);
}

void Ressource::unmap()
{	
	cudaGraphicsUnmapResources(1, &mCudaRessource, mCudaStream);
}

//TextureDeviceMemory

TextureDeviceHandle::TextureDeviceHandle(size_t width, size_t height, size_t pitch, void* linearMemory)
{
	this->width        = width;
	this->height       = height;
	this->pitch        = pitch;
	this->linearMemory = linearMemory;
	this->mCudaArray   = NULL;
}

void* TextureDeviceHandle::getPointer()
{
	return linearMemory;
}