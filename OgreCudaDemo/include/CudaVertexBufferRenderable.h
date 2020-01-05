#pragma once

#include <OgreSimpleRenderable.h>
#include <OgreHardwareBuffer.h>
#include <OgreHardwareVertexBuffer.h>

class CudaVertexBufferRenderable :  public Ogre::SimpleRenderable
{
	public:
		CudaVertexBufferRenderable(int width, int height);
		Ogre::HardwareVertexBufferSharedPtr getHardwareVertexBuffer();

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