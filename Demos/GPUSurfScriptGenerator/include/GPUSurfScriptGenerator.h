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

#include <string>
#include <vector>

struct GaussianKernel
{
	public:
		GaussianKernel(double _sigma2);

		int radius;
		double sigma2;
		std::vector<float> data;
		float norm;
};

class GPUSurfScriptGenerator
{
	public:
		GPUSurfScriptGenerator(int _width, int _height, int _nbOctave);
		void createMaterials();
		void createMaterialsHLSL();
		void createMaterialsGLSL();

	protected:
		int mNbOctave;

		int mVideoWidth;
		int mVideoHeight;

		std::string displayStepX_hlsl();
		std::string displayStepY_hlsl();
		std::string displayStepXY_hlsl();
		std::string displayStepXYHalf_hlsl();
		std::string displayGaussianKernel_hlsl();

		std::string displayStepX_glsl();
		std::string displayStepY_glsl();
		std::string displayStepXY_glsl();
		std::string displayStepXYHalf_glsl();
		std::string displayGaussianKernel_glsl();
		std::string displayCopyright();
};