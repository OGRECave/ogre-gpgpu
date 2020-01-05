#pragma once

#include <windows.h>
#include <string>
#include <iostream>

class Chrono
{	
	public:
		Chrono(bool _autostart = false);

		void start();		
		unsigned int getTimeElapsed(); //in ms

	private:
		__int64 mFreq;
		__int64 mStart;
};