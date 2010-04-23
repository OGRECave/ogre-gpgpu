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

#include "OgreCuda.h"

#include <OgreGLRenderSystem.h>
#include <OgreGLTexture.h>
#include <OgreGLHardwareVertexBuffer.h>

namespace Ogre
{
	namespace Cuda
	{
		class GLRoot : public Root
		{
			public:
				GLRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem);
				virtual void init();

			protected:
				int mDevice;
		};

		class GLTexture : public Texture
		{
			public:
				GLTexture(Ogre::TexturePtr& texture);

				virtual void registerForCudaUse();

			protected:
				GLuint mGLTextureId;
		};

		class GLVertexBuffer : public VertexBuffer
		{
			public:
				GLVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer);
				virtual void registerForCudaUse();

			protected:
				GLuint mGLVertexBufferId;
		};

		class GLTextureManager : public TextureManager
		{
			public:
				virtual Texture* createTexture(Ogre::TexturePtr texture);
				virtual void destroyTexture(Texture* texture);
		};

		class GLVertexBufferManager : public VertexBufferManager
		{
			public:
				virtual VertexBuffer* createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer);
				virtual void destroyVertexBuffer(VertexBuffer* vertexBuffer);
		};
	}
}