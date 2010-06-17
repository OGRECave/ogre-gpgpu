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

#pragma once

#include <OgreCuda.h>

#include "Feature.h"

extern "C" int extractFeatureLocationCuda(size_t width, size_t height, void* deviceTexture, 
							   CUDPPHandle& scanPlan,
							   int octave,
							   float* devicePass1, 
							   float* devicePass2, 
#ifdef GPUSURF_HOST_DEBUG
							   float* hostPass1, 
							   float* hostPass2, 
#endif
							   Feature* deviceFeatures,
							   int featureStartIndex);
extern "C" void copyCuda2Tex1D(int width, int height, void* deviceTexture, Feature* deviceFeatures, unsigned int nbFeatureFound);
extern "C" void copyTex1D2Cuda(Feature* deviceFeatures, int width, int height, void* deviceTexture, unsigned int nbFeatureFound);
