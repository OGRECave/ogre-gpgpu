#pragma once

#include <OgreRoot.h>
#include <OgreLogManager.h>

namespace Ogre
{
	class GPGPUDemo
	{
		public:
			GPGPUDemo();
			~GPGPUDemo();

			void launch();

		protected:
			Ogre::Root* mRoot;
			Ogre::Log*  mLog;
			Ogre::LogManager* mLogManager;
	};
}