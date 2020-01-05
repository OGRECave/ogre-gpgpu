#include "Chrono.h"

Chrono::Chrono(bool _autostart)
{
	QueryPerformanceFrequency((LARGE_INTEGER*)&mFreq);

	if (_autostart)
		start();
}

void Chrono::start()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&mStart);
}

unsigned int Chrono::getTimeElapsed()
{
	__int64 stop;
	QueryPerformanceCounter((LARGE_INTEGER*) &stop);
		
	return abs((int)(((stop - mStart) * 1000) / mFreq));
}