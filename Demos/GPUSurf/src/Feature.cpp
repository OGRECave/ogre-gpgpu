#include "Feature.h"

Feature::Feature(float x, float y, float scale, float octave)
{
	this->x = x;
	this->y = y;		
	this->scale  = scale;
	this->octave = octave;
	this->orientation = 0;
};

Feature::Feature()
{
	x = 0;
	y = 0;
	scale  = 0;
	octave = 0;
	orientation = 0;
};