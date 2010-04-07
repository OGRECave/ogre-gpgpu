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
	void* data = NULL;	
	renderWindow->getCustomAttribute("D3DDEVICE", &data);

	mDevice = (ID3D10Device*) data;
	mTextureManager = new Ogre::Cuda::D3D10TextureManager;
}

void D3D10Root::init()
{
	cudaD3D10SetDirect3DDevice(mDevice);
	cudaThreadSynchronize();
}

//CudaD3D10Texture

D3D10Texture::D3D10Texture(Ogre::TexturePtr& texture)
: Texture(texture)
{
	mD3D10Texture = static_cast<Ogre::D3D10TexturePtr>(mTexture)->GetTex2D();
	D3D10_TEXTURE2D_DESC desc;
	mD3D10Texture->GetDesc(&desc);
	mNbMipMaps = desc.MipLevels;
}

void D3D10Texture::registerForCudaUse()
{ 
	cudaD3D10RegisterResource(mD3D10Texture, CU_D3D10_REGISTER_FLAGS_NONE);
}

void D3D10Texture::unregister()
{
	cudaD3D10UnregisterResource(mD3D10Texture);
}

void D3D10Texture::map()
{
	cudaD3D10MapResources(1, (ID3D10Resource**)&mD3D10Texture);
}

void D3D10Texture::unmap()
{
	cudaD3D10UnmapResources(1, (ID3D10Resource**)&mD3D10Texture);
}

void* D3D10Texture::getPointer(unsigned int face, unsigned int level)
{
	void* devicePointer;
	unsigned int subResource = D3D10CalcSubresource(level, face, mNbMipMaps);
	cudaD3D10ResourceGetMappedPointer(&devicePointer, mD3D10Texture, subResource);

	return devicePointer;
}

Ogre::Vector2 D3D10Texture::getDimensions(unsigned int face, unsigned int level)
{
	size_t width, height;
	unsigned int subResource = D3D10CalcSubresource(level, face, mNbMipMaps);
	cudaD3D10ResourceGetSurfaceDimensions(&width, &height, NULL, mD3D10Texture, subResource);

	return Ogre::Vector2((Ogre::Real)width, (Ogre::Real)height);
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