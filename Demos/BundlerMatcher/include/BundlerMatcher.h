#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

#include "SiftGPU.h"

typedef std::vector<SiftGPU::SiftKeypoint> SiftKeyPoints;
typedef std::vector<float> SiftKeyDescriptors;

typedef std::pair<unsigned int, unsigned int> Match;

struct MatchInfo
{
	MatchInfo(int indexA, int indexB, std::vector<Match> matches)
	{
		this->indexA = indexA;
		this->indexB = indexB;
		this->matches = matches;
	}

	int indexA;
	int indexB;
	std::vector<Match> matches;
};

struct FeatureInfo
{
	FeatureInfo(int width, int height, SiftKeyPoints points, SiftKeyDescriptors descriptors)
	{
		this->width  = width;
		this->height = height;
		this->points = points;
		this->descriptors = descriptors;
	}

	int width;
	int height;
	SiftKeyPoints points;
	SiftKeyDescriptors descriptors;
};

class BundlerMatcher
{
	public:
		BundlerMatcher(float matchThreshold, int firstOctave = 1);
		~BundlerMatcher();
		 
		//load list.txt and output gpu.matches.txt + one key file per pictures
		void open(const std::string& inputFilename, const std::string& outMatchFilename);

	protected:
		
		//Feature extraction
		int extractSiftFeature(int fileIndex);
		bool saveKeyFile(int fileIndex);

		//Feature matching
		void matchSiftFeature(int fileIndexA, int fileIndexB);	
		void saveMatches(const std::string& filename);

		//Helpers
		bool parseListFile(const std::string& filename);
		void clearScreen();
	
		bool                     mIsInitialized;
		SiftGPU*                 mSift;
		SiftMatchGPU*            mMatcher;
		int                      mMatchBuffer[4096][2];
		float                    mMatchThreshold;

		std::vector<std::string> mFilenames;    //N images
		std::vector<FeatureInfo> mFeatureInfos; //N FeatureInfo
		std::vector<MatchInfo>   mMatchInfos;   //N(N-1)/2 MatchInfo
};