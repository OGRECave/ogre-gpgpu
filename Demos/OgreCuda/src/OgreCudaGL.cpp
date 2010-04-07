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

#include "OgreCudaGL.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaGL.h>

//GLRoot

using namespace Ogre::Cuda;

GLRoot::GLRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
: Root()
{
	void* data = NULL;
	renderWindow->getCustomAttribute("GLCONTEXT", &data);

	Ogre::GLContext* context = (GLContext*) data;

	mDevice = 0;
	mTextureManager = new Ogre::Cuda::GLTextureManager;
}

void GLRoot::init()
{
	cudaGLSetGLDevice(mDevice);
	cudaThreadSynchronize();
}

//CudaGLTexture

GLTexture::GLTexture(Ogre::TexturePtr& texture)
: Texture(texture), mDevicePointer(NULL)
{
	mGLTextureId = static_cast<Ogre::GLTexturePtr>(mTexture)->getGLID();
}

void GLTexture::registerForCudaUse()
{
	cudaGLRegisterBufferObject(mGLTextureId);
}

void GLTexture::unregister()
{
	cudaGLUnregisterBufferObject(mGLTextureId);
}

void GLTexture::map()
{
	cudaGLMapBufferObject(&mDevicePointer, mGLTextureId);
}

void GLTexture::unmap()
{	
	cudaGLUnmapBufferObject(mGLTextureId);
}

void* GLTexture::getPointer(unsigned int face, unsigned int level)
{
	//do something usefull with mDevicePointer

	return NULL;
}

Ogre::Vector2 GLTexture::getDimensions(unsigned int face, unsigned int level)
{
	//very hacky solution that doesn't take face in count
	int width  = mTexture->getWidth();
	int height = mTexture->getHeight();

	for (unsigned int i=0; i<level; ++i)
	{
		width  /= 2;
		height /= 2;
	}
	return Ogre::Vector2((Ogre::Real)width, (Ogre::Real)height);
}

//GLTextureManager

Texture* GLTextureManager::createTexture(Ogre::TexturePtr texture)
{
	return new Ogre::Cuda::GLTexture(texture);
}

void GLTextureManager::destroyTexture(Texture* texture)
{
	delete (GLTexture*)texture;
	texture = NULL;
}