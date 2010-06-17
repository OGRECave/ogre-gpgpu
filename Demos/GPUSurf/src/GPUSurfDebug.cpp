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

#include "GPUSurf.h"

#include <OgreTechnique.h>
#include <OgreMaterialManager.h>
#include <OgrePanelOverlayElement.h>
#include <OgreOverlayManager.h>

using namespace GPUSurf;

void Plan::createDebugMaterials()
{
	std::vector<std::string> textures;
	textures.push_back("GPGPU/Gray");
	textures.push_back("GPGPU/Gx");
	textures.push_back("GPGPU/Gy");
	textures.push_back("GPGPU/H");
	textures.push_back("GPGPU/NMS");

	for (int o=1; o<mNbOctave; ++o)
	{
		std::stringstream materialName;
		materialName << "GPGPU/DownSampling/Mipmap" << o;
		Ogre::MaterialPtr material = static_cast<Ogre::MaterialPtr>(Ogre::MaterialManager::getSingleton().getByName(materialName.str()));
		material.get()->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("octave", (float)o);
	}

	for (unsigned int i=0; i<textures.size(); ++i)
	{
		std::string texture = textures[i];

		for (int o=0; o<mNbOctave; ++o)
		{
			std::stringstream materialName;
			materialName << texture << "/" << o;

			Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().create(materialName.str(), Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
			Ogre::Technique *technique = material->createTechnique();
			technique->createPass();
			material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
			material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
			material->getTechnique(0)->getPass(0)->createTextureUnitState(texture);
			material->getTechnique(0)->getPass(0)->setVertexProgram("GPGPU_fixed_vp"); 
			material->getTechnique(0)->getPass(0)->setFragmentProgram("GPGPU_octave_fp");
			material->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("octave", (float)o);	
			if (texture == "GPGPU/NMS" || texture == "GPGPU/H")
				material->getTechnique(0)->getPass(0)->getFragmentProgramParameters()->setNamedConstant("scale", Ogre::Vector4(255.0f, 255.0f, 255.0f, 1.0f));
		}
	}
}

void Plan::createDebugOverlays()
{
	createDebugMaterials();

	std::vector<std::string> textures;
	textures.push_back("GPGPU/Gray");
	//textures.push_back("GPGPU/Gx");
	//textures.push_back("GPGPU/Gy");
	//textures.push_back("GPGPU/H");
	textures.push_back("GPGPU/NMS");
	 
	Ogre::OverlayManager& overlayManager = Ogre::OverlayManager::getSingleton();
	Ogre::Overlay* overlay = overlayManager.create("GPGPU/Debug/Overlay");

	for (unsigned int i=0; i<textures.size(); ++i)
	{
		std::string texture = textures[i];

		for (int o=0; o<mNbOctave; ++o)
		{
			int pot = 1 << o;
			int width = mWidth / pot;
			int height = mHeight / pot;
			int left = 0; 
			for (int j=0; j<o; ++j)
			{
				int pot = 1 << j;
				left += mWidth / pot;
			}
			int top = i*mHeight;

			std::stringstream panelName, materialName;
			panelName << texture << "/Panel/" << o;
			materialName << texture << "/" << o;

			Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", panelName.str()));
			panel->setMetricsMode(Ogre::GMM_PIXELS);
			panel->setMaterialName(materialName.str());
			panel->setDimensions((Ogre::Real)width, (Ogre::Real)height);
			panel->setPosition((Ogre::Real)left, (Ogre::Real)top);
			overlay->add2D(panel);
		}
	}
/*

	{
		Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "ReferencePicturePanel"));
		panel->setMetricsMode(Ogre::GMM_PIXELS);
		panel->setMaterialName("GPGPU/Gray/0");
		panel->setDimensions(640.0f, 480.0f);
		panel->setPosition(0.0f, 0.0f);
		overlay->add2D(panel);
	}
*/
/*
	{
		Ogre::PanelOverlayElement* panel = static_cast<Ogre::PanelOverlayElement*>(overlayManager.createOverlayElement("Panel", "FeaturesFoundPanel"));
		panel->setMetricsMode(GMM_PIXELS);
		panel->setMaterialName("FeaturesFound");
		panel->setDimensions(mVideoWidth*4, mVideoHeight*2);
		panel->setPosition(0, 0);
		overlay->add2D(panel);
	}
*/
	overlay->show();
}