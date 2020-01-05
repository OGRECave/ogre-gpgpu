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

#include <CL/OpenCL.h>

namespace Ogre
{
	namespace OpenCL
	{
		class Ressource;
		//class Texture;
		//class TextureManager;
		class VertexBufferManager;
		struct DeviceProperties;

		enum RessourceType
		{
			TEXTURE_RESSOURCE,
			VERTEXBUFFER_RESSOURCE
		};

		class Root
		{
			public:
				virtual bool init() = 0;
				void shutdown();
				void synchronize();

				//TextureManager* getTextureManager();
				VertexBufferManager* getVertexBufferManager();

				cl_command_queue* getCommandQueue();
				cl_context* getContext();

				bool map(std::vector<Ogre::OpenCL::Ressource*> ressources);   //efficient way to map multiple Ressource in one call
				bool unmap(std::vector<Ogre::OpenCL::Ressource*> ressources); //efficient way to unmap multiple Ressource in one call

				static Root* createRoot(Ogre::RenderWindow* renderWindow, Ogre::RenderSystem* renderSystem);
				static void destroyRoot(Root* root);

			protected:
				Root();
				virtual ~Root();

				//Ogre::OpenCL::TextureManager*      mTextureManager;
				Ogre::OpenCL::VertexBufferManager* mVertexBufferManager;

				cl_context       mContext;
				cl_command_queue mCommandQueue;
		};

		class Ressource
		{
			friend class Root;	

			public:
				Ressource();

				virtual bool registerForCL() = 0;
				virtual bool unregister();
				cl_mem* getPointer();

				virtual bool map() = 0;
				virtual bool unmap() = 0;

				virtual Ogre::OpenCL::RessourceType getType() = 0;

			protected:
				cl_mem mMemory;
		};

		class VertexBuffer : public Ressource
		{
			friend class VertexBufferManager;

			public:
				virtual bool registerForCL() = 0;				

				virtual bool map() = 0;
				virtual bool unmap() = 0;

				virtual Ogre::OpenCL::RessourceType getType();

			protected:
				VertexBuffer(Root* root, Ogre::HardwareVertexBufferSharedPtr vertexBuffer);

				Ogre::HardwareVertexBufferSharedPtr mVertexBuffer;
				Root* mRoot;
		};

		class VertexBufferManager
		{
			public:
				VertexBufferManager(Root* root);

				virtual VertexBuffer* createVertexBuffer(Ogre::HardwareVertexBufferSharedPtr vertexBuffer) = 0;
				virtual void destroyVertexBuffer(VertexBuffer* vertexBuffer) = 0;

			protected:
				Root* mRoot;
		};

	}
}