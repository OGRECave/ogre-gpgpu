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

#include <OgreGLHardwarePixelBuffer.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaGL.h>

#include <OgreWin32Context.h>

//GLRoot

using namespace Ogre::Cuda;

GLRoot::GLRoot(Ogre::RenderWindow* renderWindow)
: Root()
{
	Ogre::GLContext* context = NULL;
	renderWindow->getCustomAttribute("GLCONTEXT", (void*) &context);

	mDevice = 0; //this value should be extracted from Ogre (using GLContext ?)
	mTextureManager = new Ogre::Cuda::GLTextureManager;
	mVertexBufferManager = new Ogre::Cuda::GLVertexBufferManager;
}

void GLRoot::init()
{
	cudaGLSetGLDevice(mDevice);
	synchronize();
}

//CudaGLTexture

GLTexture::GLTexture(Ogre::TexturePtr& texture)
: Texture(texture)
{
	mGLTextureId = static_cast<Ogre::GLTexturePtr>(mTexture)->getGLID();
}

void GLTexture::registerForCudaUse()
{
	GLenum target = static_cast<Ogre::GLTexturePtr>(mTexture)->getGLTextureTarget();
	cudaGraphicsGLRegisterImage(&mCudaRessource, mGLTextureId, target, cudaGraphicsRegisterFlagsNone);
	allocate();
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

//GLVertexBuffer

GLVertexBuffer::GLVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: VertexBuffer(vertexBuffer)
{
	mGLVertexBufferId = static_cast<Ogre::GLHardwareVertexBuffer*>(vertexBuffer.get())->getGLBufferId();
}

void GLVertexBuffer::registerForCudaUse()
{
	cudaGraphicsGLRegisterBuffer(&mCudaRessource, mGLVertexBufferId, cudaGraphicsRegisterFlagsNone);
}

//GLVertexBufferManager

VertexBuffer* GLVertexBufferManager::createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
{
	return new Ogre::Cuda::GLVertexBuffer(vertexBuffer);
}

void GLVertexBufferManager::destroyVertexBuffer(VertexBuffer* vertexBuffer)
{
	delete (GLVertexBuffer*)vertexBuffer;
	vertexBuffer = NULL;
}