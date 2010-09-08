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

#include "BundlerMatcher.h"

#include <iostream>
#include <string>
#include <sstream>

#define GL_RGB  0x1907
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401

#include <IL/il.h>

BundlerMatcher::BundlerMatcher(float matchThreshold, int firstOctave)
{
	//DevIL init
	ilInit();
	ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
	ilEnable(IL_ORIGIN_SET);

	std::cout << "[BundlerMatcher]"<<std::endl;
	std::cout << "[Initialization]";

	mIsInitialized = true;
	mMatchThreshold = matchThreshold; //0.0 means few match and 1.0 many match

	char fo[10];
	sprintf(fo, "%d", firstOctave);
	char* args[] = {"-fo", fo};

	mSift = new SiftGPU;	
	mSift->ParseParam(2, args);	
	mSift->SetVerbose(-2);

	int support = mSift->CreateContextGL();
	if (support != SiftGPU::SIFTGPU_FULL_SUPPORTED)
		mIsInitialized = false;

	if (mIsInitialized)
		mSift->AllocatePyramid(1600, 1600);

	mMatcher = new SiftMatchGPU(4096);
	if (mMatcher->VerifyContextGL() == 0)
		mIsInitialized = false;
}

BundlerMatcher::~BundlerMatcher()
{
	//DevIL shutdown
	ilShutDown();
}

void BundlerMatcher::open(const std::string& inputFilename, const std::string& outMatchFilename)
{
	if (!mIsInitialized)
	{
		std::cout << "Error : can not initialize opengl context for SiftGPU" <<std::endl;
		return;
	}

	if (!parseListFile(inputFilename))
	{
		std::cout << "Error : can not open file : " <<inputFilename.c_str() <<std::endl;
		return;	
	}
	
	//Sift Feature Extraction
	for (unsigned int i=0; i<mFilenames.size(); ++i)
	{
		int percent = (int)(((i+1)*100.0f) / (1.0f*mFilenames.size()));		
		int nbFeature = extractSiftFeature(i);
		saveKeyFile(i);
		clearScreen();
		std::cout << "[Extracting Sift Feature : " << percent << "%] - ("<<i+1<<"/"<<mFilenames.size()<<", #"<< nbFeature <<" features)";
	}
	clearScreen();
	std::cout << "[Sift Feature extracted]"<<std::endl;
	saveVector();

	delete mSift;
	mSift = NULL;

	//Sift Matching
	int currentIteration = 0;
	int maxIterations    = (int) mFilenames.size()*((int) mFilenames.size()-1)/2; // Sum(1 -> n) = n(n-1)/2 ;-)

	for (unsigned int i=0; i<mFilenames.size(); ++i)
	{		
		for (unsigned int j=i+1; j<mFilenames.size(); ++j)
		{
			clearScreen();
			int percent = (int) (currentIteration*100.0f / maxIterations*1.0f);
			std::cout << "[Matching Sift Feature : " << percent << "%] - (" << i << "/" << j << ")";
			matchSiftFeature(i, j);
			currentIteration++;
		}
	}
	clearScreen();
	std::cout << "[Sift Feature matched]"<<std::endl;

	delete mMatcher;
	mMatcher = NULL;

	saveMatches(outMatchFilename);
	saveMatrix();
}

bool BundlerMatcher::parseListFile(const std::string& filename)
{
	std::ifstream input(filename.c_str());
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

	return true;
}

int BundlerMatcher::extractSiftFeature(int fileIndex)
{
	char* filename = &mFilenames[fileIndex][0];
	bool extracted = true;

	unsigned int imgId = 0;
	ilGenImages(1, &imgId);
	ilBindImage(imgId); 

	if(ilLoadImage(filename))
	{
		int w = ilGetInteger(IL_IMAGE_WIDTH);
		int h = ilGetInteger(IL_IMAGE_HEIGHT);
		int format = ilGetInteger(IL_IMAGE_FORMAT);

		if (mSift->RunSIFT(w, h, ilGetData(), format, GL_UNSIGNED_BYTE))
		{
			int num = mSift->GetFeatureNum();

			SiftKeyDescriptors descriptors(128*num);
			SiftKeyPoints keys(num);

			mSift->GetFeatureVector(&keys[0], &descriptors[0]);

			//Save Feature in RAM
			mFeatureInfos.push_back(FeatureInfo(w, h, keys, descriptors));

			return num;
		}
		else
		{
			extracted = false;
		}
	}
	else
	{
		extracted = false;
	}

	ilDeleteImages(1, &imgId); 

	if (!extracted)
	{
		std::cout << "Error while reading : " <<filename <<std::endl;
	}

	return -1;
}

void BundlerMatcher::matchSiftFeature(int fileIndexA, int fileIndexB)
{
	SiftKeyPoints pointsA           = mFeatureInfos[fileIndexA].points;
	SiftKeyDescriptors descriptorsA = mFeatureInfos[fileIndexA].descriptors;

	SiftKeyPoints pointsB           = mFeatureInfos[fileIndexB].points;
	SiftKeyDescriptors descriptorsB = mFeatureInfos[fileIndexB].descriptors;

	mMatcher->SetDescriptors(0, (int) pointsA.size(), &descriptorsA[0]);
	mMatcher->SetDescriptors(1, (int) pointsB.size(), &descriptorsB[0]);
	int nbMatch = mMatcher->GetSiftMatch(4096, mMatchBuffer, mMatchThreshold);

	//Save Match in RAM
	std::vector<Match> matches(nbMatch);
	for (int i=0; i<nbMatch; ++i)
		matches[i] = Match(mMatchBuffer[i][0], mMatchBuffer[i][1]);
	mMatchInfos.push_back(MatchInfo(fileIndexA, fileIndexB, matches));
}

bool BundlerMatcher::saveKeyFile(int fileIndex)
{
	std::stringstream filename;
	filename << mFilenames[fileIndex] << ".key";
	mSift->SaveSIFT(filename.str().c_str());

	return true;
}

void BundlerMatcher::clearScreen()
{
	std::cout << "\r                                                                          \r";
}

void BundlerMatcher::saveMatches(const std::string& filename)
{
	std::ofstream output;
	output.open(filename.c_str());
	for (unsigned int i=0; i<mMatchInfos.size(); ++i)
	{
		int nbMatch = (int) mMatchInfos[i].matches.size();

		output << mMatchInfos[i].indexA << " " << mMatchInfos[i].indexB << std::endl;		
		output << nbMatch <<std::endl;

		for (int j=0; j<nbMatch; ++j)
		{
			unsigned int indexA = mMatchInfos[i].matches[j].first;
			unsigned int indexB = mMatchInfos[i].matches[j].second;

			output << indexA << " " <<indexB << std::endl;
		}
	}
	output.close();
}

void BundlerMatcher::saveMatrix()
{
	std::ofstream output;
	output.open("matrix.txt");

	int nbFile = (int) mFilenames.size();

	std::vector<int> matrix(nbFile*nbFile);

	for (int i=0; i<nbFile*nbFile; ++i)
		matrix[i] = 0;

	for (unsigned int i=0; i<mMatchInfos.size(); ++i)
	{
		int indexA = mMatchInfos[i].indexA;
		int indexB = mMatchInfos[i].indexB;
		int nbMatch = (int) mMatchInfos[i].matches.size();
		matrix[indexA*nbFile+indexB] = nbMatch;
	}

	for (int i=0; i<nbFile; ++i)
	{
		for (int j=0; j<nbFile; ++j)	
		{
			output << matrix[i*nbFile+j] << ";";
		}
		output <<std::endl;
	}

	output.close();
}

void BundlerMatcher::saveVector()
{
	std::ofstream output;
	output.open("vector.txt");
	for (unsigned int i=0; i<mFeatureInfos.size(); ++i)
		output <<mFilenames[i] << ";"<<mFeatureInfos[i].points.size() << std::endl;
	output.close();
}