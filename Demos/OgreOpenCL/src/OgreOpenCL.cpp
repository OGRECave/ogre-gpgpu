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

#include "OgreOpenCL.h"
#include "OgreOpenCLDX9.h"
#include "OgreOpenCLDX10.h"
#include "OgreOpenCLGL.h"

using namespace Ogre::OpenCL;

//Root

Root::Root()
: /*mTextureManager(NULL),*/ mVertexBufferManager(NULL)
{
	mContext      = NULL;
	mCommandQueue = NULL;
}

Root::~Root()
{
	//delete mTextureManager;
	delete mVertexBufferManager;
}

Root* Root::createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
{
	std::string renderSystemName = renderSystem->getName();

	if (renderSystemName == "OpenGL Rendering Subsystem")
		return new Ogre::OpenCL::GLRoot(renderWindow);
#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
	else if (renderSystemName == "Direct3D9 Rendering Subsystem")
		return new Ogre::OpenCL::D3D9Root(renderWindow, renderSystem);
	else if (renderSystemName == "Direct3D10 Rendering Subsystem")
		return new Ogre::OpenCL::D3D10Root(renderWindow, renderSystem);
#endif
	return NULL;
}

void Root::destroyRoot(Root* root)
{
	delete root;
	root = NULL;
}

void Root::shutdown()
{
	clReleaseContext(mContext);
}

cl_command_queue* Root::getCommandQueue()
{
	return &mCommandQueue;
}

cl_context* Root::getContext()
{
	return &mContext;
}

VertexBufferManager* Root::getVertexBufferManager()
{
	return mVertexBufferManager;
}

//Ressource

Ressource::Ressource()
: mMemory(NULL)
{}

bool Ressource::unregister()
{
	cl_int error = clReleaseMemObject(mMemory);
	return error == CL_SUCCESS;
}

cl_mem* Ressource::getPointer()
{
	return &mMemory;
}

//VertexBuffer

VertexBuffer::VertexBuffer(Root* root, Ogre::HardwareVertexBufferSharedPtr vertexBuffer)
: Ressource(), mRoot(root), mVertexBuffer(vertexBuffer)
{}

Ogre::OpenCL::RessourceType VertexBuffer::getType()
{
	return Ogre::OpenCL::VERTEXBUFFER_RESSOURCE;
}

//VertexBufferManager

VertexBufferManager::VertexBufferManager(Ogre::OpenCL::Root* root)
: mRoot(root)
{}