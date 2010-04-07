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

#include "OgreCuda.h"
#include "OgreCudaD3D9.h"
#include "OgreCudaD3D10.h"
#include "OgreCudaGL.h"

#include <cuda_runtime.h>

using namespace Ogre::Cuda;

//Root

Root::Root()
: mTextureManager(NULL)
{}

Root::~Root()
{
	delete mTextureManager;
	mTextureManager = NULL;
}

void Root::shutdown()
{
	cudaThreadExit();
}

TextureManager* Root::getTextureManager()
{
	return mTextureManager;
}

Root* Root::createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
{
	std::string renderSystemName = renderSystem->getName();

	if (renderSystemName == "OpenGL Rendering Subsystem")
		return new Ogre::Cuda::GLRoot(renderWindow, renderSystem);
	else if (renderSystemName == "Direct3D9 Rendering Subsystem")
		return new Ogre::Cuda::D3D9Root(renderWindow);
	else if (renderSystemName == "Direct3D10 Rendering Subsystem")
		return new Ogre::Cuda::D3D10Root(renderWindow);
	/*
	else if (renderSystemName == "Direct3D11 Rendering Subsystem")
		return new Ogre::Cuda::D3D11Root(renderWindow);
	*/

	return NULL;
}

void Root::destroyRoot(Root* root)
{
	delete root;
	root = NULL;
}

//Texture

Texture::Texture(Ogre::TexturePtr texture)
: mTexture(texture)
{}