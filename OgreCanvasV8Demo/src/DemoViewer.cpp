#include "DemoViewer.h"

using namespace std;
DemoGroup::DemoGroup(const std::string& _name)
{
	name = _name;
}

Demo::Demo(const std::string& _reference, const std::string& _script, const std::string& _name, int _width, int _height)
{
	reference = _reference;
	script    = _script;
	name      = _name;
	width     = _width;
	height    = _height;
}

DemoViewer::DemoViewer()
{
	parseXML("../../media/demos.xml");
}

void DemoViewer::parseXML(const std::string& _name)
{
	TiXmlDocument doc(_name.c_str());
	TiXmlNode* node  = 0;
	TiXmlNode* elt   = 0;
	TiXmlNode* carto = 0;
	TiXmlNode* camera = 0;
	TiXmlNode* symbolDrawing = 0;
	
	if (doc.LoadFile())
	{
		node = doc.FirstChild("demos");
		for (TiXmlNode* elt = node->FirstChild("group"); elt; elt = elt->NextSibling("group"))
		{
			DemoGroup group = parseDemoGroup(elt);
			mDemoGroups.push_back(group);
		}
	}
}

DemoGroup DemoViewer::parseDemoGroup(TiXmlNode* _node)
{
	
	DemoGroup group(string(_node->ToElement()->Attribute("name")));

	for (TiXmlNode* elt = _node->FirstChild("demo"); elt; elt = elt->NextSibling("demo"))
	{
		Demo demo = parseDemo(elt);
		group.demos.push_back(demo);
	}
	return group;
}

Demo DemoViewer::parseDemo(TiXmlNode* _node)
{
	int width  = 150;
	int height = 150;
	string reference = string(_node->ToElement()->Attribute("reference"));
	string script    = string(_node->ToElement()->Attribute("script"));
	string name      = string(_node->ToElement()->Attribute("name"));
	_node->ToElement()->QueryIntAttribute( "width", &width);
	_node->ToElement()->QueryIntAttribute( "height", &height);

	return Demo(reference, script, name, width, height);
}

int DemoViewer::getNbDemo()
{
	int nbDemo = 0;
	for (unsigned int i=0; i<mDemoGroups.size(); ++i)
		nbDemo += mDemoGroups[i].demos.size();
	
	return nbDemo;
}

Demo DemoViewer::getDemo(int _index)
{
	int index = 0;
	for (unsigned int i=0; i<mDemoGroups.size(); ++i)
	{
		for (unsigned int j=0; j<mDemoGroups[i].demos.size(); ++j)
		{
			if (index == _index)
				return mDemoGroups[i].demos[j];
			index++;
		}
	}
	return Demo("", "", "", 150, 150);
}