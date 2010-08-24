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

#include "PhotoSynthConverter.h"

#include <PhotoSynthParser.h>
#include <PhotoSynthRadialUndistort.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
#include <OgreString.h>

using namespace PhotoSynth;
namespace bf = boost::filesystem;

bool Converter::convert(const std::string& url)
{
	std::string guid = Parser::extractGuid(url);
	if (guid == "")
	{
		std::cerr << "URL not valid (should be: http://photosynth.net/view.aspx?cid=GUID)" << std::endl;
		return false;
	}

	std::cout << "Preparing PhotoSynth " << guid << " for PMVS2" <<std::endl;

	if (!bf::exists(guid))
	{
		std::cerr << "Error: " << guid << " folder missing: you need to run PhotoSynthDownloader first !" << std::endl;
		return false;
	}

	std::stringstream path;
	path << guid << "/txt";

	if (!bf::exists(path.str()))
		bf::create_directory(path.str());

	path.str("");
	path << guid << "/visualize";
	if (!bf::exists(path.str()))
		bf::create_directory(path.str());

	path.str("");
	path << guid << "/models";
	if (!bf::exists(path.str()))
		bf::create_directory(path.str());

	path.str("");
	path << guid << "/distort";
	if (!bf::exists(path.str()))
		bf::create_directory(path.str());

	path.str("");
	path << guid << "/" << Parser::jsonFilename;
	if (!bf::exists(path.str()))
	{
		std::cerr << "Error: " << Parser::jsonFilename << " missing: you need to run PhotoSynthDownloader first !" << std::endl;
		return false;
	}

	path.str("");
	path << guid << "/" << Parser::soapFilename;
	if (!bf::exists(path.str()))
	{
		std::cerr << "Error: " << Parser::soapFilename << " missing: you need to run PhotoSynthDownloader first !" << std::endl;
		return false;
	}

	Parser parser(guid);
	parser.parseSoap();
	parser.parseJson();
	parser.parserBin();
	savePly(guid, &parser);

	if (!scanDistortFolder(guid, Parser::createFilePath(guid, "distort"), &parser))
	{
		std::cerr << "Error: " << mInputImages.size() << " images in your GUID/distort folder but " << parser.getJsonInfo().thumbs.size() << " referenced in this PhotoSynth"<< std::endl;
		return false;
	}
	
	const CoordSystem coord = parser.getCoordSystem(0);
	for (unsigned int i=0; i<coord.cameras.size(); ++i)
	{
		Camera cam = coord.cameras[i];
		int cameraIndex = cam.index;
		RadialUndistort::undistort(guid, i, mInputImages[cameraIndex], cam);
	}
	savePMVSOptions(guid, parser.getNbCamera(0));
	
	return true;
}

bool Converter::scanDistortFolder(const std::string& guid, const std::string& distortFolder, Parser* parser)
{
	bf::path path = bf::path(distortFolder);

	unsigned int counter = 0;
	bf::directory_iterator itEnd;

	//Caution: will iterate in lexico-order: thumbs_1.jpg < thumbs_10.jpg < thumbs_2.jpg
	//-> use padding to prevent this error : 00001.jpg > 00002.jpg > 00010.jpg
	for (bf::directory_iterator it(path); it != itEnd; ++it)
	{
		if (!bf::is_directory(it->status()))
		{
			std::string filename = it->filename();
			std::string extension = bf::extension(*it); //.JPG
			extension = extension.substr(1);
			Ogre::StringUtil::toLowerCase(extension); //jpg

			if (extension == "jpg")
			{	
				std::stringstream filepath;
				filepath << guid << "/distort/" << filename;				
				mInputImages.push_back(filepath.str());
				counter++;
			}
		}
	}

	if (mInputImages.size() != parser->getJsonInfo().thumbs.size())
		return false;

	return true;
}

void Converter::savePMVSOptions(const std::string& guid, unsigned int nbImage)
{
	std::ofstream output(Parser::createFilePath(guid, "pmvs_options.txt").c_str());
	if (output.is_open())
	{
		output << "level 1" << std::endl;
		output << "csize 2" << std::endl;
		output << "threshold 0.7" << std::endl;
		output << "wsize 7" << std::endl;
		output << "minImageNum 3" << std::endl;
		output << "CPU 4" << std::endl;
		output << "setEdge 0" << std::endl;
		output << "useBound 0" << std::endl;
		output << "useVisData 0" << std::endl;
		output << "sequence -1" << std::endl;
		output << "timages -1 0 "<< nbImage << std::endl;
		output << "oimages -3" << std::endl;
	}
	output.close();
}

void Converter::savePly(const std::string& guid, Parser* parser)
{
	for (unsigned int i=0; i<parser->getNbCoordSystem(); ++i)
	{		
		std::stringstream filepath;
		filepath << guid << "/bin/coord_system_" << i << ".ply";
		
		std::ofstream output(filepath.str().c_str());
		if (output.is_open())
		{
			output << "ply" << std::endl;
			output << "format ascii 1.0" << std::endl;
			output << "element vertex " << parser->getNbVertex(i) << std::endl;
			output << "property float x" << std::endl;
			output << "property float y" << std::endl;
			output << "property float z" << std::endl;
			output << "property uchar red" << std::endl;
			output << "property uchar green" << std::endl;
			output << "property uchar blue" << std::endl;
			output << "element face 0" << std::endl;
			output << "property list uchar int vertex_indices" << std::endl;
			output << "end_header" << std::endl;

			for (unsigned int j=0; j<parser->getNbPointCloud(i); ++j)
			{
				const PointCloud& pointCloud = parser->getPointCloud(i, j);
				for (unsigned int k=0; k<pointCloud.vertices.size(); ++k)
				{
					Ogre::Vector3 pos = pointCloud.vertices[k].position;
					Ogre::ColourValue color = pointCloud.vertices[k].color;
					output << pos.x << " " << pos.y << " " << pos.z << " " << (int)(color.r*255.0f) << " " << (int)(color.g*255.0f) << " " << (int)(color.b*255.0f) << std::endl;
				}				
			}
		}
		output.close();
	}
}