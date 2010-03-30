#pragma once

#include <ARToolKitPlus/TrackerMultiMarker.h>
#include <OgreMatrix4.h>
#include <vector>

struct Marker
{
	Marker();
	Marker(const Ogre::Matrix4& _trans, int _id);

	int id;
	Ogre::Matrix4 trans;
};

class TrackingSystem
{
	public:
		TrackingSystem();
		virtual ~TrackingSystem();

		void init(int _width, int _height);

		bool update(const Ogre::PixelBox& grayLevelFrame); //return true if pose is computed

		bool isPoseComputed() const;
		Ogre::Vector3 getTranslation() const;
		Ogre::Quaternion getOrientation() const;

		const std::vector<Marker> getMarkersInfo() const;
		const std::vector<int>    getVisibleMarkersId() const;

		static std::string configFilename;
		static std::string calibrationFilename;		
		static bool isUsingFullResImage;
		static bool isUsingHistory;
		static bool isUsingAutoThreshold;
		static int threshold;

	protected:		

		void convertPoseToOgreCoordinate();		
		Ogre::Matrix4 convert(const ARFloat _trans[3][4]) const;
		Ogre::Quaternion mRot180Z;
					
		ARToolKitPlus::TrackerMultiMarker *mTracker;
		bool mMarkersFound;
		bool mInitialized;

		Ogre::Vector3     mTranslation;
		Ogre::Quaternion  mOrientation;
		bool              mPoseComputed;
};
