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

#include "OgreGPGPU.h"
#include "OgreGPGPUGL.h"

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
	#include "OgreGPGPUD3D9.h"
	#include "OgreGPGPUD3D10.h"
#endif

#include <OgreRoot.h>
#include <OgreMaterialManager.h>

using namespace Ogre::GPGPU;

/************************************************************************/
/*                     Ogre::GPGPU::Root                                */
/************************************************************************/

Root::Root()
{
	mMaterialCounter = 0;
	mSceneManager = Ogre::Root::getSingleton().createSceneManager(Ogre::ST_GENERIC, "Ogre/GPGPU/SceneManager");
	mCamera = mSceneManager->createCamera("Ogre/GPGPU/Camera");
	mCamera->setUseIdentityProjection(true);
	mCamera->setUseIdentityView(true);
	mCamera->setNearClipDistance(1);

	//Get render operation from quad primitive
	mQuad = new Ogre::Rectangle2D(true);
	mQuad->setCorners(-1, 1, 1, -1);
	mQuad->getRenderOperation(mRenderOperation);
}

Root::~Root()
{
	delete mQuad;
	mSceneManager->destroyCamera(mCamera);
	Ogre::Root::getSingleton().destroySceneManager(mSceneManager);
}

void Root::compute(Result* result, Operation* operation)
{
	result->mRenderTexture->setActive(true);
	mSceneManager->manualRender(&mRenderOperation, operation->mPass, result->mViewport, Ogre::Matrix4::IDENTITY, Ogre::Matrix4::IDENTITY, Ogre::Matrix4::IDENTITY, true);
	result->mRenderTexture->setActive(false);
}

void Root::waitForCompletion()
{
	std::cout << "not available if you didn't create platform specific Ogre::GPGPU::Root"<<std::endl;
}

Root* Root::createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem)
{
	std::string renderSystemName = Ogre::Root::getSingleton().getRenderSystem()->getName();

	if (renderSystemName == "OpenGL Rendering Subsystem")
	{
		return new Ogre::GPGPU::GLRoot;
	}
#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
	else if (renderSystemName == "Direct3D9 Rendering Subsystem")
	{
		//directx query wait for completion
		return new Ogre::GPGPU::D3D9Root(renderWindow);
	}
	else if (renderSystemName == "Direct3D10 Rendering Subsystem")
	{
		//directx query wait for completion
		return new Ogre::GPGPU::D3D10Root(renderWindow);
	}
#endif
	return NULL;
}

void Root::destroyRoot(Root* root)
{
	delete root;
	root = NULL;
}

Operation* Root::createOperation(Ogre::Pass* pass)
{
	return new Operation(pass);
}

Operation* Root::createOperation(const std::string& pixelShaderName)
{
	std::stringstream name;
	name << "Ogre/GPGPU/Operation" << mMaterialCounter;
	mMaterialCounter++;

	Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().create(name.str(), Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::Pass* pass = mat->createTechnique()->createPass();
	pass->setLightingEnabled(false);
	pass->setDepthWriteEnabled(false);
	pass->setVertexProgram("GPGPU_fixed_vp");  //StdQuad
	pass->setFragmentProgram(pixelShaderName);

	return new Operation(pass);
}

Result* Root::createResult(Ogre::HardwarePixelBufferSharedPtr pixelBuffer)
{
	return new Result(pixelBuffer, mCamera);
}

void Root::destroyOperation(Operation* operation)
{
	delete operation;
	operation = NULL;
}

void Root::destroyResult(Result* result)
{
	delete result;
	result = NULL;
}

/************************************************************************/
/*                    Ogre::GPGPU::Operation                            */
/************************************************************************/

Operation::Operation(Ogre::Pass* pass)
{
	mPass = pass;
}

void Operation::setInput(Ogre::Texture* texture)
{
	Ogre::TextureUnitState* unitState = mPass->getTextureUnitState(0);
	if (!unitState)
	{
		unitState = mPass->createTextureUnitState(texture->getName());
		//unitState->setTextureFiltering(Ogre::FO_POINT, Ogre::FO_POINT, Ogre::FO_POINT);
		//unitState->setTextureAddressingMode()
	}
}

void Operation::setParameter(const std::string& name, Ogre::Real value)
{
	mPass->getFragmentProgramParameters()->setNamedConstant(name, (float)value);
}

/************************************************************************/
/*                     Ogre::GPGPU::Result                              */
/************************************************************************/	

Result::Result(Ogre::HardwarePixelBufferSharedPtr pixelBuffer, Ogre::Camera* cam)
{
	mPixelBuffer   = pixelBuffer;
	mRenderTexture = pixelBuffer->getRenderTarget();
	mRenderTexture->setAutoUpdated(false);
	mRenderTexture->setActive(false);
	mRenderTexture->update(false);
	mViewport = mRenderTexture->addViewport(cam);
	mViewport->setOverlaysEnabled(false);
}

void Result::save(const std::string &filename)
{
	size_t width  = mPixelBuffer->getWidth();
	size_t height = mPixelBuffer->getHeight();
	size_t depth  = mPixelBuffer->getDepth();
	Ogre::PixelFormat format = mPixelBuffer->getFormat();
	size_t pixelSize = Ogre::PixelUtil::getNumElemBytes(format);

	unsigned char* buffer = new unsigned char[width*height*pixelSize];

	Ogre::PixelBox box(width, height, depth, format, buffer);
	mPixelBuffer->blitToMemory(box);	

	Ogre::Image image;
	image.loadDynamicImage((Ogre::uchar*)box.data, box.getWidth(), box.getHeight(), box.getDepth(), box.format);
	image.save(filename);
	
	delete[] buffer;
}