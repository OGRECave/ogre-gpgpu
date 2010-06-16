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

#pragma once

#include <OgreGPGPUPrerequisites.h>

#include <OgreMaterial.h>
#include <OgreHardwarePixelBuffer.h>
#include <OgreCamera.h>
#include <OgreRectangle2D.h>

namespace Ogre
{
	namespace GPGPU
	{
		class Operation;
		class Result;

		class _OgreGPGPUExport Root
		{
			public:
				Root();
				virtual ~Root();

				void compute(Result* result, Operation* operation);
				virtual void waitForCompletion();

				Operation* createOperation(Ogre::Pass* pass);
				Operation* createOperation(const std::string& pixelShaderName);
				Result* createResult(Ogre::HardwarePixelBufferSharedPtr pixelBuffer);
				void destroyOperation(Operation* operation);
				void destroyResult(Result* result);

				static Root* createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem);
				static void destroyRoot(Root* root);

			protected:
				int                   mMaterialCounter;
				Ogre::Camera*         mCamera;
				Ogre::SceneManager*   mSceneManager;
				Ogre::Rectangle2D*    mQuad;
				Ogre::RenderOperation mRenderOperation;
		};

		class _OgreGPGPUExport Operation
		{
			friend class Root;	
			public:			
				Operation(Ogre::Pass* pass);
				void setInput(Ogre::Texture* texture);
				void setParameter(const std::string& name, Ogre::Real value);

			protected:
				Ogre::Pass* mPass;
		};

		class _OgreGPGPUExport Result
		{
			friend class Root;
			public:
				Result(Ogre::HardwarePixelBufferSharedPtr pixelBuffer, Ogre::Camera* cam);
				void save(const std::string& filename);

			protected:
				Ogre::HardwarePixelBufferSharedPtr mPixelBuffer;
				Ogre::RenderTexture* mRenderTexture;
				Ogre::Viewport*      mViewport;
		};
	}
}