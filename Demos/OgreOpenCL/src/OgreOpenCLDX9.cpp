#include "OgreOpenCLDX9.h"

using namespace Ogre::OpenCL;

#include <CL/cl_d3d9_ext.h>

clCreateFromD3D9VolumeTextureNV_fn clCreateFromD3D9VolumeTextureNV = (clCreateFromD3D9VolumeTextureNV_fn) clGetExtensionFunctionAddress("clCreateFromD3D9VolumeTextureNV");
clCreateFromD3D9VertexBufferNV_fn  clCreateFromD3D9VertexBufferNV  = (clCreateFromD3D9VertexBufferNV_fn)  clGetExtensionFunctionAddress("clCreateFromD3D9VertexBufferNV");
clEnqueueAcquireD3D9ObjectsNV_fn   clEnqueueAcquireD3D9ObjectsNV   = (clEnqueueAcquireD3D9ObjectsNV_fn)   clGetExtensionFunctionAddress("clEnqueueAcquireD3D9ObjectsNV");
clEnqueueReleaseD3D9ObjectsNV_fn   clEnqueueReleaseD3D9ObjectsNV   = (clEnqueueReleaseD3D9ObjectsNV_fn)   clGetExtensionFunctionAddress("clEnqueueReleaseD3D9ObjectsNV");

//D3D9Root

D3D9Root::D3D9Root(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
: Root()
{
	mGPUVendor = renderSystem->getCapabilities()->getVendor();
	renderWindow->getCustomAttribute("D3DDEVICE", (void*) &mDevice);
	//mTextureManager      = new Ogre::OpenCL::D3D9TextureManager(this);
	mVertexBufferManager = new Ogre::OpenCL::D3D9VertexBufferManager(this);
}

bool D3D9Root::init()
{
	cl_int error;

	cl_platform_id platform;
	cl_device_id device;

	//Get the number of Platform available
	cl_uint numPlatforms = 0;
	clGetPlatformIDs (0, NULL, &numPlatforms);

	if (numPlatforms == 0)
		return false;
	else
	{
		//Take the first platform available !
		//Should search for NVIDIA or AMD platform using platform name...
		clGetPlatformIDs(1, &platform, NULL);

		//Get the number of GPU devices available to the platform
		cl_uint nbGPU = 0;
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &nbGPU);

		if (nbGPU == 0)
			return false;
		else
		{
			//Take the first device available !
			//Should search a device that support graphic interop (context sharing)			
			clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		}
	}

	if (mGPUVendor != Ogre::GPU_NVIDIA)
		return false;
	else
	{
		cl_context_properties props[] = {
			CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
			CL_CONTEXT_D3D9_DEVICE_NV, (cl_context_properties)mDevice,
			0
		};

		mContext = clCreateContext(props, 1, &device, NULL, NULL, &error);
		mCommandQueue = clCreateCommandQueue(mContext, device, 0, &error);
	}

	return error == CL_SUCCESS;
}

//D3D9VertexBuffer

D3D9VertexBuffer::D3D9VertexBuffer(Root* root, Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: VertexBuffer(root, vertexBuffer)
{}

bool D3D9VertexBuffer::map()
{
	cl_int error = clEnqueueAcquireD3D9ObjectsNV(*mRoot->getCommandQueue(), 1, &mMemory, 0, 0, 0);	
	return error == CL_SUCCESS;
}

bool D3D9VertexBuffer::unmap()
{
	cl_int error = clEnqueueReleaseD3D9ObjectsNV(*mRoot->getCommandQueue(), 1, &mMemory, 0, 0, 0);
	return error == CL_SUCCESS;
}

bool D3D9VertexBuffer::registerForCL()
{
	cl_int error;
	IDirect3DVertexBuffer9* d3d9VertexBuffer = static_cast<Ogre::D3D9HardwareVertexBuffer*>(mVertexBuffer.get())->getD3D9VertexBuffer();
	mMemory = clCreateFromD3D9VertexBufferNV(*mRoot->getContext(), CL_MEM_WRITE_ONLY, d3d9VertexBuffer, &error);
	return error == CL_SUCCESS;
}

//D3D9VertexBufferManager

D3D9VertexBufferManager::D3D9VertexBufferManager(Root* root)
: VertexBufferManager(root)
{}

VertexBuffer* D3D9VertexBufferManager::createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
{
	return new D3D9VertexBuffer(mRoot, vertexBuffer);
}

void D3D9VertexBufferManager::destroyVertexBuffer(VertexBuffer* vertexBuffer)
{
	delete (D3D9VertexBuffer*) vertexBuffer;
	vertexBuffer = NULL;
}