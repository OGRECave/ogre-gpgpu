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

#include "OgreCudaD3D9.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d9_interop.h>
#include <cudaD3D9.h>

//D3D9Root

using namespace Ogre::Cuda;

D3D9Root::D3D9Root(Ogre::RenderWindow* renderWindow)
: Root()
{
	renderWindow->getCustomAttribute("D3DDEVICE", (void*) &mDevice);
	mTextureManager      = new Ogre::Cuda::D3D9TextureManager;
	mVertexBufferManager = new Ogre::Cuda::D3D9VertexBufferManager;
}

void D3D9Root::init()
{
	cudaD3D9SetDirect3DDevice(mDevice);
	wait();
}

//D3D9Texture

D3D9Texture::D3D9Texture(Ogre::TexturePtr& texture)
: Texture(texture)
{
	mD3D9Texture = static_cast<Ogre::D3D9TexturePtr>(mTexture)->getTexture();
}

void D3D9Texture::registerForCudaUse()
{	
	cudaGraphicsD3D9RegisterResource(&mCudaRessource, (LPDIRECT3DRESOURCE9)mD3D9Texture, cudaGraphicsRegisterFlagsNone);
	allocate();
}

//D3D9TextureManager

Texture* D3D9TextureManager::createTexture(Ogre::TexturePtr texture)
{
	return new Ogre::Cuda::D3D9Texture(texture);
}

void D3D9TextureManager::destroyTexture(Texture* texture)
{
	delete (D3D9Texture*)texture;
	texture = NULL;
}

//D3D9VertexBuffer

D3D9VertexBuffer::D3D9VertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: VertexBuffer(vertexBuffer)
{
	mD3D9VertexBuffer = static_cast<Ogre::D3D9HardwareVertexBuffer*>(vertexBuffer.get())->getD3D9VertexBuffer();
}

void D3D9VertexBuffer::registerForCudaUse()
{
	cudaGraphicsD3D9RegisterResource(&mCudaRessource, (LPDIRECT3DRESOURCE9)mD3D9VertexBuffer, cudaGraphicsRegisterFlagsNone);
}

//D3D9VertexBufferManager

VertexBuffer* D3D9VertexBufferManager::createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
{
	return new Ogre::Cuda::D3D9VertexBuffer(vertexBuffer);
}

void D3D9VertexBufferManager::destroyVertexBuffer(VertexBuffer* vertexBuffer)
{
	delete (D3D9VertexBuffer*)vertexBuffer;
	vertexBuffer = NULL;
}