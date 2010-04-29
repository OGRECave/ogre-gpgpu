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

#include "OgreCudaD3D10.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d10_interop.h>
#include <cudaD3D10.h>

//D3D10Root

using namespace Ogre::Cuda;

D3D10Root::D3D10Root(Ogre::RenderWindow* renderWindow)
: Root()
{
	renderWindow->getCustomAttribute("D3DDEVICE", (void*) &mDevice);
	mTextureManager      = new Ogre::Cuda::D3D10TextureManager;
	mVertexBufferManager = NULL; //new Ogre::Cuda::D3D10VertexBufferManager;
}

void D3D10Root::init()
{
	cudaD3D10SetDirect3DDevice(mDevice);
	synchronize();
}

//CudaD3D10Texture

D3D10Texture::D3D10Texture(Ogre::TexturePtr& texture)
: Texture(texture)
{
	mD3D10Texture = static_cast<Ogre::D3D10TexturePtr>(mTexture)->getTextureResource();
}

void D3D10Texture::registerForCudaUse()
{ 
	cudaGraphicsD3D10RegisterResource(&mCudaRessource, mD3D10Texture, cudaGraphicsRegisterFlagsNone);
	allocate();
}

//D3D10TextureManager

Texture* D3D10TextureManager::createTexture(Ogre::TexturePtr texture)
{
	return new Ogre::Cuda::D3D10Texture(texture);
}

void D3D10TextureManager::destroyTexture(Texture* texture)
{
	delete (D3D10Texture*)texture;
	texture = NULL;
}

//D3D10VertexBuffer

D3D10VertexBuffer::D3D10VertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: VertexBuffer(vertexBuffer)
{
	mD3D10VertexBuffer = NULL;
	//mD3D10VertexBuffer = static_cast<Ogre::D3D10HardwareVertexBuffer*>(vertexBuffer.get())->getD3DVertexBuffer();
}

void D3D10VertexBuffer::registerForCudaUse()
{
	//cudaGraphicsD3D10RegisterResource(&mCudaRessource, (ID3D10Resource*)mD3D10VertexBuffer, cudaGraphicsRegisterFlagsNone);
}

//D3D10VertexBufferManager

VertexBuffer* D3D10VertexBufferManager::createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
{
	return NULL; //new Ogre::Cuda::D3D10VertexBuffer(vertexBuffer);
}

void D3D10VertexBufferManager::destroyVertexBuffer(VertexBuffer* vertexBuffer)
{
	delete (D3D10VertexBuffer*)vertexBuffer;
	vertexBuffer = NULL;
}