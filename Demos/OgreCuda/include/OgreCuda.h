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

#include <OgreVector2.h>
#include <OgreTexture.h>
#include <OgreRenderSystem.h>
#include <OgreRenderWindow.h>

namespace Ogre
{
	namespace Cuda
	{
		class TextureManager;

		//TODO : Error handling + Check device capability
		class Root
		{
			public:
				virtual void init() = 0;
				void shutdown();
				TextureManager* getTextureManager();

				static Root* createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem);
				static void destroyRoot(Root* root);

			protected:
				Root();
				virtual ~Root();

				Ogre::Cuda::TextureManager* mTextureManager;
		};

		class Texture
		{
			friend class TextureManager;	

			public:
				virtual void registerForCudaUse() = 0;
				virtual void unregister() = 0;

				virtual void map() = 0;
				virtual void unmap() = 0;
				virtual void* getPointer(unsigned int face, unsigned int level) = 0;
				virtual Ogre::Vector2 getDimensions(unsigned int face, unsigned int level) = 0;

			protected:
				Texture(Ogre::TexturePtr texture);

				Ogre::TexturePtr mTexture;		
		};

		class TextureManager
		{
			public:
				virtual Texture* createTexture(Ogre::TexturePtr texture) = 0;
				virtual void destroyTexture(Texture* texture) = 0;
		};
	}
}