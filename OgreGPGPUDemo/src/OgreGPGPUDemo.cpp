#include "OgreGPGPUDemo.h"

#include <OgreMaterialManager.h>
#include <OgreGPGPU.h>

#include "Chrono.h"

using namespace Ogre;

GPGPUDemo::GPGPUDemo()
{
	mLogManager = new Ogre::LogManager();
	mLog  = mLogManager->createLog("Ogre.log", true, false);
	mRoot = new Ogre::Root("plugins.cfg", "ogre.cfg");
	Ogre::LogManager::getSingleton().getDefaultLog()->setDebugOutputEnabled(false);

	mRoot->showConfigDialog();
	mRoot->initialise(true, "Ogre/GPGPU/DemoApplication");

	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/GPGPUDemo", "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/StdQuad",   "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
}

GPGPUDemo::~GPGPUDemo()
{
	mRoot->shutdown();
	delete mRoot;
}

void GPGPUDemo::launch()
{
	// create rendertarget
	Ogre::TexturePtr tex = Ogre::TextureManager::getSingleton().createManual("Ogre/GPGPU/RT", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D, 512, 512, 0, Ogre::PF_R8G8B8A8, Ogre::TU_RENDERTARGET);

	// load material
	Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().getByName("GPGPUDemo");
	mat->load();

	Ogre::GPGPU::Root* gpgpu = new Ogre::GPGPU::Root;
	Ogre::GPGPU::Result* result = gpgpu->createResult(tex->getBuffer(0, 0));
	Ogre::GPGPU::Operation* op  = gpgpu->createOperation(mat->getTechnique(0)->getPass(0));

	Chrono chrono(true);
	for (unsigned int i=0; i<5000; ++i)
		gpgpu->compute(result, op);
	result->save("gpgpu_computing.png");
	std::cout << chrono.getTimeElapsed() << "ms [rendering+saving]" << std::endl;
}