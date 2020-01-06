#pragma once

#include <OgreSimpleRenderable.h>
#include <OgreHardwareBuffer.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreMaterialManager.h>

class CudaVertexBufferRenderable :  public Ogre::SimpleRenderable
{
	public:
		CudaVertexBufferRenderable(int width, int height);
		Ogre::HardwareVertexBufferSharedPtr getHardwareVertexBuffer();

        void setMaterial(const Ogre::String& name) { Ogre::SimpleRenderable::setMaterial(Ogre::MaterialManager::getSingleton().getByName(name)); }

	protected:
		void createMaterial();
		void fillHardwareBuffers();
		virtual Ogre::Real getBoundingRadius(void) const;
		virtual Ogre::Real getSquaredViewDepth(const Ogre::Camera *) const;		

	protected:
		int mWidth;
		int mHeight;
		Ogre::HardwareVertexBufferSharedPtr mVertexBuffer;
};