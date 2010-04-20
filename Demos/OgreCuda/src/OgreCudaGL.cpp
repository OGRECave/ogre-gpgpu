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

//GLRoot

using namespace Ogre::Cuda;

GLRoot::GLRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
: Root()
{
	void* data = NULL;
	renderWindow->getCustomAttribute("GLCONTEXT", &data);

	Ogre::GLContext* context = (GLContext*) data;

	mDevice = 0; //this value should be extracted from Ogre (using GLContext ?)
	mTextureManager = new Ogre::Cuda::GLTextureManager;
}

void GLRoot::init()
{
	cudaGLSetGLDevice(mDevice);
	wait();
}

//CudaGLTexture

GLTexture::GLTexture(Ogre::TexturePtr& texture)
: Texture(texture)
{
	mGLTextureId = static_cast<Ogre::GLTexturePtr>(mTexture)->getGLID();
}

void GLTexture::registerForCudaUse()
{
	cudaGraphicsGLRegisterImage(&mCudaRessource, mGLTextureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	cudaMallocPitch(&mCudaLinearMemory, &mPitch, mTexture->getWidth() * sizeof(char) * 4, mTexture->getHeight());
	cudaMemset(mCudaLinearMemory, 1, mPitch * mTexture->getHeight());
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