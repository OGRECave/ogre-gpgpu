#include "OgreOpenCLGL.h"

#include <OgreWin32Context.h>
#include <OgreGLHardwareVertexBuffer.h>

using namespace Ogre::OpenCL;

//GLRoot

GLRoot::GLRoot(Ogre::RenderWindow* renderWindow)
: Root()
{
	Ogre::GLContext* context = NULL;
	renderWindow->getCustomAttribute("GLCONTEXT", (void*) &context);

	mDevice = 0; //this value should be extracted from Ogre (using GLContext ?)
	//mTextureManager = new Ogre::Cuda::GLTextureManager;
	mVertexBufferManager = new Ogre::OpenCL::GLVertexBufferManager(this);
}

bool GLRoot::init()
{
	cl_int error;

#ifdef __APPLE__

	CGLContextObj kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
	cl_context_properties props[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 
		0 
	};
	mContext = clCreateContext(props, 0,0, NULL, NULL, &error);

#else

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

	#ifdef UNIX
		
		cl_context_properties props[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
			CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
			CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 
			0
		};
		mContext = clCreateContext(props, 1, &device, NULL, NULL, &error);

	#else //WIN32

		cl_context_properties props[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
			CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 
			0
		};
		mContext = clCreateContext(props, 1, &device, NULL, NULL, &error);

	#endif

#endif

	// create a command queue for first device the context reported 
	mCommandQueue = clCreateCommandQueue(mContext, device, 0, &error);

	return error == CL_SUCCESS;
}

//GLVertexBuffer

GLVertexBuffer::GLVertexBuffer(Root* root, Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: VertexBuffer(root, vertexBuffer)
{}

bool GLVertexBuffer::map()
{
	glFinish(); //TODO : check if it is necessary
	cl_int error = clEnqueueAcquireGLObjects(*mRoot->getCommandQueue(), 1, &mMemory, 0,0,0);
	return error == CL_SUCCESS;
}

bool GLVertexBuffer::unmap()
{
	cl_int error = clEnqueueReleaseGLObjects(*mRoot->getCommandQueue(), 1, &mMemory, 0,0,0);
	return error == CL_SUCCESS;
}

bool GLVertexBuffer::registerForCL()
{
	cl_int error;
	unsigned int bufferId = static_cast<Ogre::GLHardwareVertexBuffer*>(mVertexBuffer.get())->getGLBufferId();	
	mMemory = clCreateFromGLBuffer(*mRoot->getContext(), CL_MEM_WRITE_ONLY, bufferId, &error);
	return error == CL_SUCCESS;
}

//GLVertexBufferManager

GLVertexBufferManager::GLVertexBufferManager(Root* root)
: VertexBufferManager(root)
{}

VertexBuffer* GLVertexBufferManager::createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
{
	return new GLVertexBuffer(mRoot, vertexBuffer);
}

void GLVertexBufferManager::destroyVertexBuffer(VertexBuffer* vertexBuffer)
{
	delete (GLVertexBuffer*) vertexBuffer;
	vertexBuffer = NULL;
}