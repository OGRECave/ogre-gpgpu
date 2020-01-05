#pragma once

#include <string>
#include <vector>
#include "tinyxml.h"

struct Demo 
{
	Demo(const std::string& _reference, const std::string& _script, const std::string& _name, int _width, int _height);

	std::string reference;
	std::string script;
	std::string name;
	int width;
	int height;
};

struct DemoGroup
{
	DemoGroup(const std::string& _name);

	std::string name;
	std::vector<Demo> demos;
};

class DemoViewer
{
	public:
		DemoViewer();

		int getNbDemo();
		Demo getDemo(int _index);

	protected:
		void parseXML(const std::string& _name);
		DemoGroup DemoViewer::parseDemoGroup(TiXmlNode* _node);
		Demo DemoViewer::parseDemo(TiXmlNode* _node);

		std::vector<DemoGroup> mDemoGroups;
};