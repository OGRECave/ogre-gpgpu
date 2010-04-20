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

#include <cuda_runtime.h>

namespace Ogre
{
	namespace Cuda
	{
		class TextureManager;
		struct DeviceProperties;

		class Root
		{
			public:
				virtual void init() = 0;
				void shutdown();
				void wait();    //synchronize Cuda thread with calling thread (wait for last Cuda call completion)

				bool isCudaStatusOK();             //check cuda status and save it
				std::string getErrorMessage();     //return a message corresponding to the last call of isCudaStatusOK()
				static std::string getLastError(); //check cuda status and return corresponding message

				TextureManager* getTextureManager();

				static Root* createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem);
				static void destroyRoot(Root* root);

				static int getDeviceCount(); //return nb device compatible with Cuda on this machine
				static DeviceProperties getDeviceProperties(int index);

				static int getCudaRuntimeVersion();
				static int getVideoDriverVersion();

			protected:
				Root();
				virtual ~Root();

				Ogre::Cuda::TextureManager* mTextureManager;
				cudaError_t mLastCudaError;
		};

		class Texture
		{
			friend class TextureManager;	

			public:
				virtual void registerForCudaUse() = 0;
				virtual void unregister();

				virtual void map();
				virtual void unmap();
				virtual void update();
				virtual void* getPointer(unsigned int face, unsigned int level);
				virtual Ogre::Vector2 getDimensions(unsigned int face, unsigned int level);

			protected:
				Texture(Ogre::TexturePtr texture);

				Ogre::TexturePtr mTexture;
				struct cudaGraphicsResource* mCudaRessource;
				void*                        mCudaLinearMemory;
				cudaStream_t 	             mCudaStream;
				cudaArray*                   mCudaArray;
				size_t                       mPitch;
		};

		class TextureManager
		{
			public:
				TextureManager();

				virtual Texture* createTexture(Ogre::TexturePtr texture) = 0;
				virtual void destroyTexture(Texture* texture) = 0;

				void map(std::vector<Ogre::Cuda::Texture*> textures);
				void unmap(std::vector<Ogre::Cuda::Texture*> textures);

			protected:
				cudaStream_t mCudaStream;
		};

		struct DeviceProperties
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
std::ostream& operator <<(std::ostream& output, const Ogre::Cuda::DeviceProperties& prop);