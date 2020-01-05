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

#include "GPUSurfPictureDetector.h"

#include <OgreRoot.h>
#include <IL/il.h>

GPUSurfPictureDetector::GPUSurfPictureDetector()
{
	mWidth   = 0;
	mHeight  = 0;
	mBuffer  = NULL;
	mImageId = 0;

	ilInit();
	ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
	ilEnable(IL_ORIGIN_SET);

	mRoot = new Ogre::Root("plugins.cfg");
	mRoot->showConfigDialog();
	mRoot->initialise(true, "Ogre/GPGPU/GPUSurfPictureDetector");

	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/StdQuad", "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/gpusurf", "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/gpusurf/hlsl",   "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/gpusurf/glsl",   "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().addResourceLocation("../../../Media/gpusurf/script", "FileSystem", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
	
	mCudaRoot = Ogre::Cuda::Root::createRoot(mRoot->getAutoCreatedWindow(), mRoot->getRenderSystem());
	mCudaRoot->init();
	mDetector = new GPUSurfFeatureDetector(mCudaRoot, 4);
}

GPUSurfPictureDetector::~GPUSurfPictureDetector()
{
	delete mDetector;
	Ogre::Cuda::Root::destroyRoot(mCudaRoot);

	//DevIL shutdown
	ilShutDown();

	mBuffer = NULL;
	
	mRoot->shutdown();
	delete mRoot;
}

unsigned int GPUSurfPictureDetector::open(const std::string& filelist)
{
	std::ifstream input(filelist.c_str());
	if (input.is_open())
	{
		while(!input.eof())
		{
			std::string line;
			std::getline(input, line);
			if (line != "")
				mFilenames.push_back(line);
		}
	}
	input.close();

	return mFilenames.size();
}

std::vector<Feature> GPUSurfPictureDetector::extract(unsigned int index)
{
	std::vector<Feature> features;
	std::string filename = mFilenames[index];
	if (openImage(filename))
	{
		std::cout << "["<<index<<"] ("<<mWidth<<"x"<<mHeight<<") " << filename << std::endl;
		Ogre::PixelBox frame(mWidth, mHeight, 1, Ogre::PF_B8G8R8, mBuffer);

		mDetector->update(frame);
		/*
		std::stringstream name;
		name << filename << ".png";
		Ogre::Image image;
		image.loadDynamicImage((Ogre::uchar*)frame.data, frame.getWidth(), frame.getHeight(), frame.format);
		image.save(name.str());
		*/

		for (unsigned int i=0; i<mDetector->getNbFeatureFound(); ++i)
		{
			Feature* f = &(mDetector->getFeatures()[i]);
			features.push_back(Feature(f->x, f->y, f->scale, f->octave));
		}

		closeImage();
	}
	return features;
}

bool GPUSurfPictureDetector::saveToXml(const std::vector<Feature>& features, unsigned int index)
{
	Ogre::String base, ext, path;
	Ogre::StringUtil::splitFullFilename(mFilenames[index], base, ext, path);
	std::stringstream filename;
	filename << path << base << ".xml";

	std::ofstream output;
	output.open(filename.str().c_str());
	bool opened = output.is_open();
	if (opened)
	{
		output << "<features nb=\""<< features.size()<<"\">" <<std::endl;
		for (unsigned int i=0; i<features.size(); ++i)
		{
			Feature f = features[i];
			output << "<f x=\""<< f.x<<"\" y=\""<<f.y<<"\" s=\"" << f.scale << "\" o=\"" << f.octave << "\" />"<< std::endl;
		}
		output << "</features>"<<std::endl;
	}
	output.close();

	return opened;
}

bool GPUSurfPictureDetector::openImage(const std::string& filename)
{
	ilGenImages(1, &mImageId);
	ilBindImage(mImageId);

	bool loaded = ilLoadImage(filename.c_str()) == IL_TRUE;
	if (loaded)
	{
		mWidth  = ilGetInteger(IL_IMAGE_WIDTH);
		mHeight = ilGetInteger(IL_IMAGE_HEIGHT);
		mBuffer = (unsigned char*) ilGetData();
	}
	return loaded;
}

void GPUSurfPictureDetector::closeImage()
{
	ilDeleteImages(1, &mImageId); 
}