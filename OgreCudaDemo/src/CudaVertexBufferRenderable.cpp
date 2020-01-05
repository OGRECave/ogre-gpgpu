#include "CudaVertexBufferRenderable.h"

#include <OgreHardwareBuffer.h>
#include <OgreHardwareBufferManager.h>
#include <OgreTechnique.h>
#include <OgreMaterialManager.h>

CudaVertexBufferRenderable::CudaVertexBufferRenderable(int width, int height)
{
	mWidth  = width;
	mHeight = height;

	// Initialize render operation
	mRenderOp.operationType = Ogre::RenderOperation::OT_POINT_LIST;
	mRenderOp.useIndexes = false;
	mRenderOp.vertexData = new Ogre::VertexData;

	// Create vertex declaration
	Ogre::VertexDeclaration *decl = mRenderOp.vertexData->vertexDeclaration;
	size_t offset = 0;
	offset += decl->addElement(0, offset, Ogre::VET_FLOAT3, Ogre::VES_POSITION).getSize();
	offset += decl->addElement(0, offset, Ogre::VET_COLOUR, Ogre::VES_DIFFUSE).getSize();

	// Make capacity the next power of two
	size_t size = mWidth*mHeight;
	size_t capacity = 1;
	while (capacity < size)
		capacity <<= 1;
	
	// Create HardwareVertexBuffer
	mVertexBuffer = Ogre::HardwareBufferManager::getSingleton().createVertexBuffer(
		mRenderOp.vertexData->vertexDeclaration->getVertexSize(0), 
		capacity, 
		Ogre::HardwareBuffer::HBU_DISCARDABLE);
	
	// Bind buffer
	mRenderOp.vertexData->vertexBufferBinding->setBinding(0, mVertexBuffer);

	// Update vertex count in the render operation
	mRenderOp.vertexData->vertexCount = size;

	fillHardwareBuffers();
	createMaterial();
	setMaterial("CudaVertexBufferMaterial");
}

void CudaVertexBufferRenderable::fillHardwareBuffers()
{
	Ogre::Real *prPos = static_cast<Ogre::Real*>(mVertexBuffer->lock(Ogre::HardwareBuffer::HBL_DISCARD));
	{
		for (int i=0; i<mWidth; ++i)
		{
			for (int j=0; j<mHeight; ++j)
			{
				*prPos++ = i*1.0f; //x
				*prPos++ = j*1.0f; //y
				*prPos++ = i*1.0f; //z
				*prPos++ = 0.0;    //color packed in float
			}
		}
	}
	mVertexBuffer->unlock();
	mBox.setInfinite();
}

Ogre::HardwareVertexBufferSharedPtr CudaVertexBufferRenderable::getHardwareVertexBuffer()
{
	return mVertexBuffer;
}

Ogre::Real CudaVertexBufferRenderable::getBoundingRadius(void) const
{
	return 0;
}

Ogre::Real CudaVertexBufferRenderable::getSquaredViewDepth(const Ogre::Camera *) const
{
	return 0;
}

void CudaVertexBufferRenderable::createMaterial()
{
	Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().create("CudaVertexBufferMaterial", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::Technique *technique = material->createTechnique();
	technique->createPass();
	material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
	material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
	material->getTechnique(0)->getPass(0)->setVertexColourTracking(Ogre::TVC_DIFFUSE);
}