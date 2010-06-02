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

#include <OgreVector2.h>
#include <OgreTexture.h>
#include <OgreRenderSystem.h>
#include <OgreRenderWindow.h>
#include "OgreCudaPrerequisites.h"

#include <cuda_runtime.h>

namespace Ogre
{
	namespace Cuda
	{
		class Ressource;
		class Texture;
		class TextureManager;
		class VertexBufferManager;		
		struct DeviceProperties;
		
		enum RessourceType
		{
			TEXTURE_RESSOURCE,
			VERTEXBUFFER_RESSOURCE
		};

		class _OgreCudaExport Root
		{
			public:
				virtual void init() = 0;
				void shutdown();
				void synchronize();    //synchronize Cuda thread with calling thread (wait for last Cuda call completion)

				TextureManager* getTextureManager();           //return TextureManager to create Ogre::Cuda::Texture
				VertexBufferManager* getVertexBufferManager(); //return VertexBufferManager to create/destroy Ogre::Cuda::VertexBuffer

				void map(std::vector<Ogre::Cuda::Ressource*> ressources);   //efficient way to map multiple Ressource in one call
				void unmap(std::vector<Ogre::Cuda::Ressource*> ressources); //efficient way to unmap multiple Ressource in one call

				bool isCudaStatusOK();             //check cuda status and save it
				std::string getErrorMessage();     //return a message corresponding to the last call of isCudaStatusOK()
				
				//Static methods :

				static std::string getLastError(); //check cuda status and return corresponding message

				static Root* createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem);
				static void destroyRoot(Root* root);

				static int getDeviceCount();                            //return how many devices are compatible with Cuda on this machine
				static DeviceProperties getDeviceProperties(int index); //return properties of the selected device

				static int getCudaRuntimeVersion();
				static int getVideoDriverVersion();

			protected:
				Root();
				virtual ~Root();

				Ogre::Cuda::TextureManager*      mTextureManager;
				Ogre::Cuda::VertexBufferManager* mVertexBufferManager;
				cudaError_t                      mLastCudaError;
				cudaStream_t                     mCudaStream;
		};

		class _OgreCudaExport Ressource
		{
			friend class Root;	

			public:
				Ressource();

				virtual void registerForCudaUse() = 0;
				virtual void unregister();

				virtual void map();
				virtual void unmap();

				virtual Ogre::Cuda::RessourceType getType() = 0;

			protected:
				struct cudaGraphicsResource*        mCudaRessource;
				cudaStream_t 	                    mCudaStream;
		};

		class _OgreCudaExport TextureDeviceHandle
		{
			friend class Ogre::Cuda::Texture;

			public:
				TextureDeviceHandle(size_t width, size_t height, size_t pitch, void* linearMemory);
				void* getPointer();

				size_t width;
				size_t height;
				size_t pitch;
				void* linearMemory;

			protected:
				cudaArray* mCudaArray;
		};

		class _OgreCudaExport Texture : public Ressource
		{
			friend class TextureManager;	

			public:
				virtual void registerForCudaUse() = 0;
				virtual void unregister();
				void updateReading(TextureDeviceHandle& mem);
				void updateWriting(TextureDeviceHandle& mem);

				TextureDeviceHandle getDeviceHandle(unsigned int face, unsigned int mipmap);
				Ogre::Vector2 getDimensions(unsigned int face, unsigned int mipmap);

				virtual Ogre::Cuda::RessourceType getType();

			protected:
				Texture(Ogre::TexturePtr texture);
				void allocate();
				unsigned int getIndex(unsigned int face, unsigned int mipmap);

				int                              mPixelSizeInBytes;
				Ogre::TexturePtr                 mTexture;
				std::vector<TextureDeviceHandle> mDevicePtrs;
		};

		class _OgreCudaExport VertexBuffer : public Ressource
		{
			friend class VertexBufferManager;

			public:
				virtual void registerForCudaUse() = 0;
				void* getPointer();

				virtual Ogre::Cuda::RessourceType getType();

			protected:
				VertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer);

				Ogre::HardwareVertexBufferSharedPtr mVertexBuffer;
				cudaArray*                          mCudaArray;
		};

		class _OgreCudaExport TextureManager
		{
			public:
				virtual Texture* createTexture(Ogre::TexturePtr texture) = 0;
				virtual void destroyTexture(Texture* texture) = 0;				
		};

		class _OgreCudaExport VertexBufferManager
		{
			public:
				virtual VertexBuffer* createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer) = 0;
				virtual void destroyVertexBuffer(VertexBuffer* vertexBuffer) = 0;
		};

		struct _OgreCudaExport DeviceProperties
		{
			DeviceProperties();
			DeviceProperties(const cudaDeviceProp& prop);

			std::string name;                 // ASCII string identifying device
			size_t totalGlobalMem;            // Global memory available on device in bytes
			size_t sharedMemPerBlock;         // Shared memory available per block in bytes
			int    regsPerBlock;              // 32-bit registers available per block
			int    warpSize;                  // Warp size in threads
			size_t memPitch;                  // Maximum pitch in bytes allowed by memory copies
			int    maxThreadsPerBlock;        // Maximum number of threads per block
			int    maxThreadsDim[3];          // Maximum size of each dimension of a block
			int    maxGridSize[3];            // Maximum size of each dimension of a grid
			int    clockRate;                 // Clock frequency in kilohertz
			size_t totalConstMem;             // Constant memory available on device in bytes
			int    major;                     // Major compute capability
			int    minor;                     // Minor compute capability
			size_t textureAlignment;          // Alignment requirement for textures
			int    deviceOverlap;             // Device can concurrently copy memory and execute a kernel
			int    multiProcessorCount;       // Number of multiprocessors on device
			int    kernelExecTimeoutEnabled;  // Specified whether there is a run time limit on kernels
			int    integrated;                // Device is integrated as opposed to discrete
			bool   canMapHostMemory;          // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
			int    computeMode;               // Compute mode (See ::cudaComputeMode)
		};
	}
}
_OgreCudaExport std::ostream& operator <<(std::ostream& output, const Ogre::Cuda::DeviceProperties& prop);