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

#include "GpuSurfScriptGenerator.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;

#define round(x) floor(x+0.5)

GaussianKernel::GaussianKernel(double _sigma2)
{
	sigma2 = _sigma2;
	radius = (int)round(5.0*sqrt(_sigma2));
	for (int r=-radius;r<=radius;r++)
		data.push_back((float)exp(-(r*r)/(2.0*_sigma2)));
	norm = 0;
	for (unsigned int i=0; i<data.size(); ++i)
		norm += data[i];
}

GPUSurfScriptGenerator::GPUSurfScriptGenerator(int _width, int _height, int _nbOctave)
{
	mNbOctave    = _nbOctave;
	mVideoWidth  = _width;
	mVideoHeight = _height;
}

void GPUSurfScriptGenerator::createMaterials()
{
	ofstream output;
//GPGPU_generated.compositor
	output.open("GPGPU_generated.compositor");
	output << displayCopyright();
	for (int o=0; o<mNbOctave; ++o)
	{		
		output << "//	MIPMAP	"<<o<<endl;
		output <<endl;

		if (o != 0)
		{
			output << "compositor GPGPU/DownSampling/Mipmap"<<o<<endl;
			output << "{"<<endl;
			output << "    technique"<<endl;
			output << "    {"<<endl;
			output << "        target_output"<<endl;
			output << "        {"<<endl;
			output << "            input none"<<endl;
			output << "            "<<endl;
			output << "            pass render_quad"<<endl;
			output << "            {"<<endl;
			output << "                material GPGPU/DownSampling/Mipmap"<<o<<endl;
			output << "            }"<<endl;
			output << "        }"<<endl;
			output << "    }"<<endl;
			output << "}"<<endl;
			output <<endl;
		}
		else
		{
			output << "compositor GPGPU/RGB2Gray/Mipmap0"<<endl;
			output << "{"<<endl;
			output << "	technique"<<endl;
			output << "	{"<<endl;
			output << "		target_output"<<endl;
			output << "		{"<<endl;
			output << "			input none"<<endl;
			output << "         "<<endl;
			output << "			pass render_quad"<<endl;
			output << "			{"<<endl;
			output << "				material GPGPU/RGB2Gray/Mipmap0"<<endl;
			output << "			}"<<endl;
			output << "		}"<<endl;
			output << "	}"<<endl;
			output << "}"<<endl;
			output <<endl;
		}
		output << "compositor GPGPU/GaussianX/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "    technique"<<endl;
		output << "    {"<<endl;
		output << "        target_output"<<endl;
		output << "        {"<<endl;
		output << "            // Start with clear output"<<endl;
		output << "            input none"<<endl;
		output << "            "<<endl;
		output << "            // Draw a fullscreen quad with the black and white image"<<endl;
		output << "            pass render_quad"<<endl;
		output << "            {"<<endl;
		output << "                // Renders a fullscreen quad with a material"<<endl;
		output << "                material GPGPU/GaussianX/Mipmap"<<o<<endl;
		output << "            }"<<endl;
		output << "        }"<<endl;
		output << "    }"<<endl;
		output << "}"<<endl;
		output<<endl;
		output << "compositor GPGPU/GaussianY/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "    technique"<<endl;
		output << "    {"<<endl;
		output << "        target_output"<<endl;
		output << "        {"<<endl;
		output << "            // Start with clear output"<<endl;
		output << "            input none"<<endl;
		output << "            "<<endl;
		output << "            // Draw a fullscreen quad with the black and white image"<<endl;
		output << "            pass render_quad"<<endl;
		output << "            {"<<endl;
		output << "                // Renders a fullscreen quad with a material"<<endl;
		output << "                material GPGPU/GaussianY/Mipmap"<<o<<endl;
		output << "            }"<<endl;
		output << "        }"<<endl;
		output << "    }"<<endl;
		output << "}"<<endl;
		output<<endl;
		output << "compositor GPGPU/Hessian/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "    technique"<<endl;
		output << "    {"<<endl;
		output << "        target_output"<<endl;
		output << "        {"<<endl;
		output << "            // Start with clear output"<<endl;
		output << "            input none"<<endl;
		output << "            "<<endl;
		output << "            // Draw a fullscreen quad with the black and white image"<<endl;
		output << "            pass render_quad"<<endl;
		output << "            {"<<endl;
		output << "                // Renders a fullscreen quad with a material"<<endl;
		output << "                material GPGPU/Hessian/Mipmap"<<o<<endl;
		output << "            }"<<endl;
		output << "        }"<<endl;
		output << "    }"<<endl;
		output << "}"<<endl;
		output<<endl;
		output << "compositor GPGPU/NMS/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "    technique"<<endl;
		output << "    {"<<endl;
		output << "        target_output"<<endl;
		output << "        {"<<endl;
		output << "            // Start with clear output"<<endl;
		output << "            input none"<<endl;
		output << "            "<<endl;
		output << "            // Draw a fullscreen quad with the black and white image"<<endl;
		output << "            pass render_quad"<<endl;
		output << "            {"<<endl;
		output << "                // Renders a fullscreen quad with a material"<<endl;
		output << "                material GPGPU/NMS/Mipmap"<<o<<endl;
		output << "            }"<<endl;
		output << "        }"<<endl;
		output << "    }"<<endl;
		output << "}"<<endl;
		output<<endl;
	}
	output.close();

//GPGPU_generated.material	
	output.open("GPGPU_generated.material");
	output << displayCopyright();
	for (int o=0; o<mNbOctave; ++o)
	{
		output << "//	MIPMAP	"<<o<<endl;
		output <<endl;

		if (o != 0)
		{
			output << "material GPGPU/DownSampling/Mipmap"<<o<<endl;
			output << "{"<<endl;
			output << "	technique"<<endl;
			output << "	{"<<endl;
			output << "		pass"<<endl;
			output << "		{"<<endl;
			output << "			lighting off"<<endl;
			output << "			depth_write off"<<endl;
			output << ""<<endl;
			output << "			vertex_program_ref Ogre/Compositor/StdQuad_Cg_vp"<<endl;
			output << "			{}"<<endl;
			output << ""<<endl;
			output << "			fragment_program_ref GPGPU_downsampling_fp"<<endl;
			output << "			{"<<endl;
			output << "				param_named octave float "<< o<<".0"<<endl;
			output << "			}"<<endl;
			output << ""<<endl;
			output << "			texture_unit"<<endl;
			output << "			{"<<endl;
			output << "				tex_address_mode clamp"<<endl;
			output << "				texture GPGPU/Gy"<<endl;
			output << "			}"<<endl;
			output << "		}"<<endl;
			output << "	}"<<endl;
			output << "}"<<endl;
			output <<endl;
		}
		else
		{
			output << "material GPGPU/RGB2Gray/Mipmap0"<<endl;
			output << "{"<<endl;
			output << "	technique"<<endl;
			output << "	{"<<endl;
			output << "		pass"<<endl;
			output << "		{"<<endl;
			output << "			lighting off"<<endl;
			output << "			depth_write off"<<endl;
			output << ""<<endl;
			output << "			vertex_program_ref Ogre/Compositor/StdQuad_Cg_vp"<<endl;
			output << "			{}"<<endl;
			output << ""<<endl;
			output << "			fragment_program_ref GPGPU_rgb2gray_fp"<<endl;
			output << "			{}"<<endl;
			output << ""<<endl;
			output << "			texture_unit"<<endl;
			output << "			{"<<endl;
			output << "				tex_address_mode clamp"<<endl;
			output << "				texture WebcamVideoTexture"<<endl;
			output << "			}"<<endl;
			output << "		}"<<endl;
			output << "	}"<<endl;
			output << "}"<<endl;
			output <<endl;
		}

		output << "material GPGPU/GaussianX/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "	technique"<<endl;
		output << "	{"<<endl;
		output << "		pass"<<endl;
		output << "		{"<<endl;
		output << "			lighting off"<<endl;
		output << "			depth_write off"<<endl;
		output << ""<<endl;
		output << "			vertex_program_ref Ogre/Compositor/StdQuad_Cg_vp"<<endl;
		output << "			{}"<<endl;
		output << ""<<endl;
		output << "			fragment_program_ref GPGPU_gaussian_x_fp"<<endl;
		output << "			{"<<endl;
		output << "				param_named octave float "<< o<<".0"<<endl;
		output << "			}"<<endl;
		output << ""<<endl;
		output << "			texture_unit"<<endl;
		output << "			{"<<endl;
		output << "				tex_border_colour 1.0 0.0 0.0 1.0"<<endl;
		output << "				tex_address_mode wrap"<<endl;
		output << "				texture GPGPU/Gray"<<endl;
		output << "			}"<<endl;
		output << "		}"<<endl;
		output << "	}"<<endl;
		output << "}"<<endl;
		output <<endl;
		output << "material GPGPU/GaussianY/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "	technique"<<endl;
		output << "	{"<<endl;
		output << "		pass"<<endl;
		output << "		{"<<endl;
		output << "			lighting off"<<endl;
		output << "			depth_write off"<<endl;
		output << ""<<endl;
		output << "			vertex_program_ref Ogre/Compositor/StdQuad_Cg_vp"<<endl;
		output << "			{}"<<endl;
		output << ""<<endl;
		output << "			fragment_program_ref GPGPU_gaussian_y_fp"<<endl;
		output << "			{"<<endl;
		output << "				param_named octave float "<< o<<".0"<<endl;
		output << "			}"<<endl;
		output << ""<<endl;
		output << "			texture_unit"<<endl;
		output << "			{"<<endl;
		output << "				tex_address_mode clamp"<<endl;
		output << "				texture GPGPU/Gx"<<endl;
		output << "			}"<<endl;
		output << "		}"<<endl;
		output << "	}"<<endl;
		output << "}"<<endl;
		output<<endl;
		output << "material GPGPU/Hessian/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "	technique"<<endl;
		output << "	{"<<endl;
		output << "		pass"<<endl;
		output << "		{"<<endl;
		output << "			lighting off"<<endl;
		output << "			depth_write off"<<endl;
		output << ""<<endl;
		output << "			vertex_program_ref Ogre/Compositor/StdQuad_Cg_vp"<<endl;
		output << "			{}"<<endl;
		output << ""<<endl;
		output << "			fragment_program_ref GPGPU_hessian_fp"<<endl;
		output << "			{"<<endl;
		output << "				param_named octave float "<< o<<".0"<<endl;
		output << "			}"<<endl;
		output << ""<<endl;
		output << "			texture_unit"<<endl;
		output << "			{"<<endl;
		output << "				tex_address_mode clamp"<<endl;
		output << "				texture GPGPU/Gy"<<endl;
		output << "			}"<<endl;
		output << "		}"<<endl;
		output << "	}"<<endl;
		output << "}"<<endl;
		output<<endl;
		output << "material GPGPU/NMS/Mipmap"<<o<<endl;
		output << "{"<<endl;
		output << "	technique"<<endl;
		output << "	{"<<endl;
		output << "		pass"<<endl;
		output << "		{"<<endl;
		output << "			lighting off"<<endl;
		output << "			depth_write off"<<endl;
		output << ""<<endl;
		output << "			vertex_program_ref Ogre/Compositor/StdQuad_Cg_vp"<<endl;
		output << "			{}"<<endl;
		output << ""<<endl;
	if (o == 0) 
	{
		output << "			fragment_program_ref GPGPU_nms_first_fp"<<endl;
	}
	else 
	{
		output << "			fragment_program_ref GPGPU_nms_other_fp"<<endl;
	}
		output << "			{"<<endl;
		output << "				param_named octave float "<< o<<".0"<<endl;
		//output << "				param_named threshold float 0.05"<<endl;
		output << "			}"<<endl;
		output << ""<<endl;
		output << "			texture_unit"<<endl;
		output << "			{"<<endl;
		output << "				filtering none"<<endl;
		output << "				tex_address_mode clamp"<<endl;
		output << "				texture GPGPU/H"<<endl;
		output << "			}"<<endl;
		output << "		}"<<endl;
		output << "	}"<<endl;
		output << "}"<<endl;
		output<<endl;
	}
	output.close();
}

void GPUSurfScriptGenerator::createMaterialsHLSL()
{
	ofstream output;

//GPGPU_gaussian_x_fp.hlsl
	output.open("GPGPU_gaussian_x_fp.hlsl");
	output << displayCopyright();
	output << "void GPGPU_gaussian_x_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepX_hlsl()<<endl;
	output << "	const float stepX = steps[octave];"<<endl;
	output << "	"<<displayGaussianKernel_hlsl()<<endl;
	output << "	float4 values = float4(0, 0, 0, 0);"<<endl;
	output << "    for (int i=0; i<19; i++) {"<<endl;
	output << "	   float4 coord = float4(uv.x+(i-9)*stepX, uv.y, 0, octave);"<<endl;
	output << "	   float luminance = tex2Dlod(tex0, coord).r;"<<endl;
	output << "	   values += kernel[i]*luminance;"<<endl;
	output << "    }"<<endl;
	output << "	result = values;"<<endl;
	output << "}"<<endl;
	output.close();	

//GPGPU_gaussian_y_fp.hlsl
	output.open("GPGPU_gaussian_y_fp.hlsl");
	output << displayCopyright();
	output << "void GPGPU_gaussian_y_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepY_hlsl()<<endl;
	output << "	const float stepY = steps[octave];"<<endl;
	output << "	"<<displayGaussianKernel_hlsl()<<endl;
	output << "	float4 values = float4(0, 0, 0, 0);"<<endl;
	output << "    for (int i=0; i<19; i++) {"<<endl;
	output << "	   float4 coord = float4(uv.x, uv.y+(i-9)*stepY, 0, octave);"<<endl;
	output << "	   float luminance = tex2Dlod(tex0, coord).r;"<<endl;
	output << "	   values += kernel[i]*luminance;"<<endl;
	output << "    }"<<endl;
	output << "	result = values;"<<endl;
	output << "}"<<endl;
	output.close();

//GPGPU_hessian_fp.hlsl
	output.open("GPGPU_hessian_fp.hlsl");
	output << displayCopyright();
	output << "void GPGPU_hessian_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepXY_hlsl()<<endl;
	output << "	const float stepX = steps[octave].x;"<<endl;
	output << "	const float stepY = steps[octave].y;"<<endl;
	output << ""<<endl;
	output << "	float4 coord = float4(uv.x, uv.y, 0, octave);"<<endl;
	output << ""<<endl;
	
	//Second-order derivative:
	// Lxx(x,y) = 1/4*(L(x+2,y) - 2*L(x,y) + L(x-2,y))
	// Lyy(x,y) = 1/4*(L(x,y+2) - 2*L(x,y) + L(x,y-2))
	// Lxy(x,y) = 1/4*(L(x-1,y-1) + L(x+1,y+1) - L(x+1,y-1) - L(x-1,y+1))

	output << "	float4 Lxx = -2*tex2Dlod(tex0, coord);"<<endl;
	output << "	float4 Lyy = Lxx;"<<endl;
	output << "	float4 Lxy;"<<endl;

	output << "	Lxx += tex2Dlod(tex0, coord+float4(-2*stepX,0,0,0)) + tex2Dlod(tex0, coord+float4(2*stepX,0,0,0));"<<endl;
	output << "	Lyy += tex2Dlod(tex0, coord+float4(0,-2*stepY,0,0))    + tex2Dlod(tex0, coord+float4(0,2*stepY,0,0));"<<endl;

	output << "	Lxy  = tex2Dlod(tex0, coord+float4(-stepX,-stepY,0,0)) + tex2Dlod(tex0, coord+float4(stepX,stepY,0,0));"<<endl;
	output << "	Lxy -= tex2Dlod(tex0, coord+float4(stepX,-stepY,0,0))  + tex2Dlod(tex0, coord+float4(-stepX,stepY,0,0));"<<endl;

	output << "	Lxx /= 4;"<<endl;
	output << "	Lyy /= 4;"<<endl;
	output << "	Lxy /= 4;"<<endl;

	output << ""<<endl;
	output << "	result = Lxx*Lyy-Lxy*Lxy;"<<endl;
	output << "}"<<endl;
	output <<endl;
	output.close();

//GPGPU_nms_first_fp.hlsl
	output.open("GPGPU_nms_first_fp.hlsl");
	output << displayCopyright();
	output << "void GPGPU_nms_first_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	uniform float threshold,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepXY_hlsl()<<endl;
	output << "	"<<endl;
	output << "	float stepX = steps[octave].x;"<<endl;
	output << "	float stepY = steps[octave].y;"<<endl;
	output << "	float4 coord = float4(uv.x-1.5*stepX, uv.y-1.5*stepY, 0, octave);"<<endl;
	output << "	"<<endl;
	output << "	float4 T0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, 0));"<<endl;
	output << "	float4 T3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, 0));"<<endl;
	output << ""<<endl;
	output << "	float4 T4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, 0)));"<<endl;
	output << "	float4 T5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, 0)));"<<endl;
	output << "	"<<endl;
	output << "	float4 R0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, 0)), T3);"<<endl;
	output << "	float4 R1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, 0)), T2);"<<endl;
	output << "	float4 R2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, 0)), T1);"<<endl;
	output << "	float4 R3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, 0)), T0);"<<endl;
	output << ""<<endl;
	output << "	float4 T8 = max(T0, T3);"<<endl;
	output << "	float4 T9 = max(T1, T2);"<<endl;
	output << "	"<<endl;
	output << "	//R0"<<endl;
	output << "	R0 = max(R0, T4);"<<endl;
	output << "	R0 = max(R0, T5);"<<endl;
	output << "	R0 = max(R0, T9);"<<endl;
	output << "				"<<endl;
	output << "	R0.xyz = max(R0.xyz, R0.yzw);"<<endl;
	output << "	R0.yzw = max(R0.xyz, R0.yzw);"<<endl;
	output << "	"<<endl;
	output << "	R0.xyz = max(R0.xyz, T0.yzw);"<<endl;
	output << "	R0.yzw = max(T0.xyz, R0.yzw); "<<endl;
	output << "	T4 = (T0>=max(R0, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//R1"<<endl;
	output << "	R1 = max(R1, T4);"<<endl;
	output << "	R1 = max(R1, T6);"<<endl;
	output << "	R1 = max(R1, T8);"<<endl;
	output << "	"<<endl;
	output << "	R1.xyz = max(R1.xyz, R1.yzw);"<<endl;
	output << "	R1.yzw = max(R1.xyz, R1.yzw);"<<endl;
	output << ""<<endl;
	output << "	R1.xyz = max(R1.xyz, T1.yzw);"<<endl;
	output << "	R1.yzw = max(T1.xyz, R1.yzw);	"<<endl;
	output << "	T5 = (T1> max(R1, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//R2"<<endl;
	output << "	R2 = max(R2, T5);"<<endl;
	output << "	R2 = max(R2, T7);"<<endl;
	output << "	R2 = max(R2, T8);"<<endl;
	output << "	"<<endl;
	output << "	R2.xyz = max(R2.xyz, R2.yzw);"<<endl;
	output << "	R2.yzw = max(R2.xyz, R2.yzw);"<<endl;
	output << "	"<<endl;
	output << "	R2.xyz = max(R2.xyz, T2.yzw);"<<endl;
	output << "	R2.yzw = max(T2.xyz, R2.yzw);	"<<endl;
	output << "	T6 = (T2> max(R2, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//R3"<<endl;
	output << "	R3 = max(R3, T6);"<<endl;
	output << "	R3 = max(R3, T7);"<<endl;
	output << "	R3 = max(R3, T9);"<<endl;
	output << "	"<<endl;
	output << "	R3.xyz = max(R3.xyz, R3.yzw);"<<endl;
	output << "	R3.yzw = max(R3.xyz, R3.yzw);"<<endl;
	output << "	"<<endl;
	output << "	R3.xyz = max(R3.xyz, T3.yzw);"<<endl;
	output << "	R3.yzw = max(T3.xyz, R3.yzw);	"<<endl;
	output << "	T7 = (T3> max(R3, threshold));"<<endl;
	output << "	"<<endl;
	output << "	float4 pixval;	"<<endl;
	output << "	pixval.x = dot(float4(0, 4, 8, 0), clamp(T5+T7, 0, 1));"<<endl;
	output << "	pixval.y = dot(float4(0, 4, 8, 0), clamp(T6+T7, 0, 1));"<<endl;
	output << "	pixval.z = dot(float4(0, 4, 8, 0), clamp(T4+T5+T6+T7, 0, 1));"<<endl;
	output << "	pixval.w = octave;"<<endl;
	output << ""<<endl;
	output << "	result = pixval / 255.0;"<<endl;
	output << "}"<<endl;
	output << ""<<endl;
	output.close();

//GPGPU_nms_other_fp.hlsl
	output.open("GPGPU_nms_other_fp.hlsl");
	output << displayCopyright();
	output << "void GPGPU_nms_other_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	uniform float threshold,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepXY_hlsl()<<endl;
	output << "	"<<endl;
	output << "	//Octave o"<<endl;
	output << "	float stepX = steps[octave].x;"<<endl;
	output << "	float stepY = steps[octave].y;"<<endl;
	output << "	"<<endl;
	output << "	float4 coord = float4(uv.x-1.5*stepX, uv.y-1.5*stepY, 0, octave);"<<endl;
	output << "	"<<endl;
	output << "	float4 T0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, 0));"<<endl;
	output << "	float4 T3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, 0));"<<endl;
	output << ""<<endl;
	output << "	float4 T4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, 0)));"<<endl;
	output << "	float4 T5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, 0)));"<<endl;
	output << "	"<<endl;
	output << "	float4 R0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, 0)), T3);"<<endl;
	output << "	float4 R1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, 0)), T2);"<<endl;
	output << "	float4 R2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, 0)), T1);"<<endl;
	output << "	float4 R3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, 0)), T0);"<<endl;
	output << ""<<endl;
	output << "	float4 T8 = max(T0, T3);"<<endl;
	output << "	float4 T9 = max(T1, T2);"<<endl;
	output << "	"<<endl;
	output << "	//R0"<<endl;
	output << "	R0 = max(R0, T4);"<<endl;
	output << "	R0 = max(R0, T5);"<<endl;
	output << "	R0 = max(R0, T9);"<<endl;
	output << "	"<<endl;
	output << "	//R1"<<endl;
	output << "	R1 = max(R1, T4);"<<endl;
	output << "	R1 = max(R1, T6);"<<endl;
	output << "	R1 = max(R1, T8);"<<endl;
	output << "	"<<endl;
	output << "	//R2"<<endl;
	output << "	R2 = max(R2, T5);"<<endl;
	output << "	R2 = max(R2, T7);"<<endl;
	output << "	R2 = max(R2, T8);"<<endl;
	output << "	"<<endl;
	output << "	//R3"<<endl;
	output << "	R3 = max(R3, T6);"<<endl;
	output << "	R3 = max(R3, T7);"<<endl;
	output << "	R3 = max(R3, T9);"<<endl;
	output << "	"<<endl;
	output << "	//Octave o-1"<<endl;
	output << "	float2 L0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, -1)).zw;"<<endl;
	output << "	float2 L1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, -1)).zw;"<<endl;
	output << "	float2 L2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, -1)).zw;"<<endl;
	output << "	float2 L3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, -1)).zw;"<<endl;
	output << ""<<endl;
	output << "	float2 L4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, -1)).zw, tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, -1)).zw);"<<endl;
	output << "	float2 L5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, -1)).zw);"<<endl;
	output << "	float2 L6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, -1)).zw);"<<endl;
	output << "	float2 L7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, -1)).zw);"<<endl;
	output << "	"<<endl;
	output << "	float2 K0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, -1)).zw, L3);"<<endl;
	output << "	float2 K1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, -1)).zw, L2);"<<endl;
	output << "	float2 K2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, -1)).zw, L1);"<<endl;
	output << "	float2 K3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, -1)).zw, L0);"<<endl;
	output << ""<<endl;
	output << "	float2 L8 = max(L0, L3);"<<endl;
	output << "	float2 L9 = max(L1, L2);	"<<endl;
	output << "	"<<endl;
	output << "	//K0"<<endl;
	output << "	K0 = max(K0, L4);"<<endl;
	output << "	K0 = max(K0, L5);"<<endl;
	output << "	K0 = max(K0, L9);"<<endl;
	output << "	"<<endl;
	output << "	//K1"<<endl;
	output << "	K1 = max(K1, L4);"<<endl;
	output << "	K1 = max(K1, L6);"<<endl;
	output << "	K1 = max(K1, L8);"<<endl;
	output << "	"<<endl;
	output << "	//K2"<<endl;
	output << "	K2 = max(K2, L5);"<<endl;
	output << "	K2 = max(K2, L7);"<<endl;
	output << "	K2 = max(K2, L8);"<<endl;
	output << "		"<<endl;
	output << "	//K3"<<endl;
	output << "	K3 = max(K3, L6);"<<endl;
	output << "	K3 = max(K3, L7);"<<endl;
	output << "	K3 = max(K3, L9);"<<endl;
	output << "	"<<endl;
	output << "	//maximum between octaves o and o-1"<<endl;
	output << "	"<<endl;
	output << "	//T0 - K0"<<endl;
	output << "	T4 = max(float4(K0.y,  R0.xyz), R0);"<<endl;
	output << "	T4 = max(float4(K0.xy, R0.xy),  T4);"<<endl;
	output << "	"<<endl;
	output << "	T4 = max(T4, T0);"<<endl;
	output << "	T4 = max(T4, float4(L0.xy, T0.xy));"<<endl;
	output << "	T4 = (float4(L0.y, T0.xyz)>=max(T4, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//T1 - K1"<<endl;
	output << "	T5 = max(float4(K1.y,  R1.xyz), R1);"<<endl;
	output << "	T5 = max(float4(K1.xy, R1.xy),  T5);"<<endl;
	output << "	"<<endl;
	output << "	T5 = max(T5, T1);"<<endl;
	output << "	T5 = max(T5, float4(L1.xy, T1.xy));"<<endl;
	output << "	T5 = (float4(L1.y, T1.xyz)>=max(T5, threshold));"<<endl;
	output << "		"<<endl;
	output << "	//T2 - K2"<<endl;
	output << "	T6 = max(float4(K2.y,  R2.xyz), R2);"<<endl;
	output << "	T6 = max(float4(K2.xy, R2.xy),  T6);"<<endl;
	output << "	"<<endl;
	output << "	T6 = max(T6, T2);"<<endl;
	output << "	T6 = max(T6, float4(L2.xy, T2.xy));"<<endl;
	output << "	T6 = (float4(L2.y, T2.xyz)>=max(T6, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//T3 - K3"<<endl;
	output << "	T7 = max(float4(K3.y,  R3.xyz), R3);"<<endl;
	output << "	T7 = max(float4(K3.xy, R3.xy),  T7);"<<endl;
	output << "	"<<endl;
	output << "	T7 = max(T7, T3);"<<endl;
	output << "	T7 = max(T7, float4(L3.xy, T3.xy));"<<endl;
	output << "	T7 = (float4(L3.y, T3.xyz)>=max(T7, threshold));"<<endl;
	output << "	"<<endl;
	output << "	float4 pixval;	"<<endl;
	output << "	pixval.x = dot(float4(1, 2, 4, 8), clamp(T5+T7, 0, 1));"<<endl;
	output << "	pixval.y = dot(float4(1, 2, 4, 8), clamp(T6+T7, 0, 1));"<<endl;
	output << "	pixval.z = dot(float4(1, 2, 4, 8), clamp(T4+T5+T6+T7, 0, 1));"<<endl;
	output << "	pixval.w = octave;"<<endl;
	output << ""<<endl;
	output << "	result = pixval / 255.0;"<<endl;
	output << "}"<<endl;
	output.close();
}

void GPUSurfScriptGenerator::createMaterialsGLSL()
{
	ofstream output;

//GPGPU_gaussian_x_fp.glsl
	output.open("GPGPU_gaussian_x_fp.glsl");
	output << displayCopyright();
	output << "void GPGPU_gaussian_x_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepX_glsl()<<endl;
	output << "	const float stepX = steps[octave];"<<endl;
	output << "	"<<displayGaussianKernel_glsl()<<endl;
	output << "	float4 values = float4(0, 0, 0, 0);"<<endl;
	output << "    for (int i=0; i<19; i++) {"<<endl;
	output << "	   float4 coord = float4(uv.x+(i-9)*stepX, uv.y, 0, octave);"<<endl;
	output << "	   float luminance = tex2Dlod(tex0, coord).r;"<<endl;
	output << "	   values += kernel[i]*luminance;"<<endl;
	output << "    }"<<endl;
	output << "	result = values;"<<endl;
	output << "}"<<endl;
	output <<endl;
	output.close();

//GPGPU_gaussian_y_fp.glsl
	output.open("GPGPU_gaussian_y_fp.glsl");
	output << displayCopyright();
	output << "void GPGPU_gaussian_y_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepY_glsl()<<endl;
	output << "	const float stepY = steps[octave];"<<endl;
	output << "	"<<displayGaussianKernel_glsl()<<endl;
	output << "	float4 values = float4(0, 0, 0, 0);"<<endl;
	output << "    for (int i=0; i<19; i++) {"<<endl;
	output << "	   float4 coord = float4(uv.x, uv.y+(i-9)*stepY, 0, octave);"<<endl;
	output << "	   float luminance = tex2D(tex0, coord).r;"<<endl;
	output << "	   values += kernel[i]*luminance;"<<endl;
	output << "    }"<<endl;
	output << "	result = values;"<<endl;
	output << "}"<<endl;
	output.close();

//GPGPU_hessian_fp.glsl
	output.open("GPGPU_hessian_fp.glsl");
	output << displayCopyright();
	output << "void GPGPU_hessian_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepXY_glsl()<<endl;
	output << "	const float stepX = steps[octave].x;"<<endl;
	output << "	const float stepY = steps[octave].y;"<<endl;
	output << ""<<endl;
	output << "	float4 coord = float4(uv.x, uv.y, 0, octave);"<<endl;
	output << ""<<endl;
/*	
	//reverse-engineering from gpusurf implementation
	// Lxx(x,y) = 2*L(x,y) - L(x-2,y) - L(x+2,y)
	// Lyy(x,y) = 2*L(x,y) - L(x,y-2) - L(x,y+2)
	// Lxy(x,y) = L(x-1,y-1) + L(x+1,y+1) - L(x+1,y-1) - L(x-1,y+1)

	//by johan ;-)
	// Lxx(x,y) = 1/4*(L(x+2,y) - 2*L(x,y) + L(x-2,y))
	// Lyy(x,y) = 1/4*(L(x,y+2) - 2*L(x,y) + L(x,y-2))
	// Lxy(x,y) = 1/4*(L(x-1,y-1) + L(x+1,y+1) - L(x+1,y-1) - L(x-1,y+1))

	output << "	float4 Lxx = 2*tex2Dlod(tex0, coord);"<<endl;
	output << "	float4 Lyy = Lxx;"<<endl;
	output << "	float4 Lxy;"<<endl;
	output << "	Lxx -= tex2Dlod(tex0, coord+float4(-2*stepX,0,0,0))    + tex2Dlod(tex0, coord+float4(2*stepX,0,0,0));"<<endl;
	output << "	Lyy -= tex2Dlod(tex0, coord+float4(0,-2*stepY,0,0))    + tex2Dlod(tex0, coord+float4(0,2*stepY,0,0));"<<endl;
	output << "	Lxy  = tex2Dlod(tex0, coord+float4(-stepX,-stepY,0,0)) + tex2Dlod(tex0, coord+float4(stepX,stepY,0,0));"<<endl;
	output << "	Lxy -= tex2Dlod(tex0, coord+float4(stepX,-stepY,0,0))  + tex2Dlod(tex0, coord+float4(-stepX,stepY,0,0));"<<endl;
*/
/*
	//http://en.wikipedia.org/wiki/Edge_detection
	// Lxx(x,y) = L(x-1,y) - 2L(x,y) + L(x+1,y)
	// Lyy(x,y) = L(x,y-1) - 2L(x,y) + L(x, y+1)
	// Lxy(x,y) = (L(x-1,y-1) - L(x-1,y+1) - L(x+1,y-1) + L(x+1,y+1))/4
*/
	output << "	float4 Lxx = -2*tex2Dlod(tex0, coord);"<<endl;
	output << "	float4 Lyy = Lxx;"<<endl;
	output << "	Lxx += tex2Dlod(tex0, coord+float4(-stepX,0,0,0))    + tex2Dlod(tex0, coord+float4(stepX,0,0,0));"<<endl;
	output << "	Lyy += tex2Dlod(tex0, coord+float4(0,-stepY,0,0))    + tex2Dlod(tex0, coord+float4(0,stepY,0,0));"<<endl;
	output << "	"<<endl;
	output << "	float4 Lxy;"<<endl;
	output << "	Lxy  = tex2Dlod(tex0, coord+float4(-stepX,-stepY,0,0)) + tex2Dlod(tex0, coord+float4(stepX,stepY,0,0));"<<endl;
	output << "	Lxy -= tex2Dlod(tex0, coord+float4(stepX,-stepY,0,0))  + tex2Dlod(tex0, coord+float4(-stepX,stepY,0,0));"<<endl;
	output << "	Lxy /= 4;"<<endl;

	output << ""<<endl;
	output << "	result = Lxx*Lyy-Lxy*Lxy;"<<endl;
	output << ""<<endl;
	output << "	//remove lines below (just for visual debugging)"<<endl;
	output << "	result.g *= 250;"<<endl;
	output << "	result.a  = 1.0;"<<endl;
	output << "}"<<endl;
	output <<endl;
	output.close();

//GPGPU_nms_first_fp.glsl
	output.open("GPGPU_nms_first_fp.glsl");
	output << displayCopyright();
	output << "void GPGPU_nms_first_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	uniform float threshold,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepXY_glsl()<<endl;
	output << "	"<<endl;
	output << "	float stepX = steps[octave].x;"<<endl;
	output << "	float stepY = steps[octave].y;"<<endl;
	output << "	float4 coord = float4(uv.x-1.5*stepX, uv.y-1.5*stepY, 0, octave);"<<endl;
	output << "	"<<endl;
	output << "	float4 T0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, 0));"<<endl;
	output << "	float4 T3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, 0));"<<endl;
	output << ""<<endl;
	output << "	float4 T4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, 0)));"<<endl;
	output << "	float4 T5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, 0)));"<<endl;
	output << "	"<<endl;
	output << "	float4 R0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, 0)), T3);"<<endl;
	output << "	float4 R1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, 0)), T2);"<<endl;
	output << "	float4 R2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, 0)), T1);"<<endl;
	output << "	float4 R3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, 0)), T0);"<<endl;
	output << ""<<endl;
	output << "	float4 T8 = max(T0, T3);"<<endl;
	output << "	float4 T9 = max(T1, T2);"<<endl;
	output << "	"<<endl;
	output << "	//R0"<<endl;
	output << "	R0 = max(R0, T4);"<<endl;
	output << "	R0 = max(R0, T5);"<<endl;
	output << "	R0 = max(R0, T9);"<<endl;
	output << "				"<<endl;
	output << "	R0.xyz = max(R0.xyz, R0.yzw);"<<endl;
	output << "	R0.yzw = max(R0.xyz, R0.yzw);"<<endl;
	output << "	"<<endl;
	output << "	R0.xyz = max(R0.xyz, T0.yzw);"<<endl;
	output << "	R0.yzw = max(T0.xyz, R0.yzw); "<<endl;
	output << "	T4 = (T0>=max(R0, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//R1"<<endl;
	output << "	R1 = max(R1, T4);"<<endl;
	output << "	R1 = max(R1, T6);"<<endl;
	output << "	R1 = max(R1, T8);"<<endl;
	output << "	"<<endl;
	output << "	R1.xyz = max(R1.xyz, R1.yzw);"<<endl;
	output << "	R1.yzw = max(R1.xyz, R1.yzw);"<<endl;
	output << ""<<endl;
	output << "	R1.xyz = max(R1.xyz, T1.yzw);"<<endl;
	output << "	R1.yzw = max(T1.xyz, R1.yzw);	"<<endl;
	output << "	T5 = (T1> max(R1, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//R2"<<endl;
	output << "	R2 = max(R2, T5);"<<endl;
	output << "	R2 = max(R2, T7);"<<endl;
	output << "	R2 = max(R2, T8);"<<endl;
	output << "	"<<endl;
	output << "	R2.xyz = max(R2.xyz, R2.yzw);"<<endl;
	output << "	R2.yzw = max(R2.xyz, R2.yzw);"<<endl;
	output << "	"<<endl;
	output << "	R2.xyz = max(R2.xyz, T2.yzw);"<<endl;
	output << "	R2.yzw = max(T2.xyz, R2.yzw);	"<<endl;
	output << "	T6 = (T2> max(R2, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//R3"<<endl;
	output << "	R3 = max(R3, T6);"<<endl;
	output << "	R3 = max(R3, T7);"<<endl;
	output << "	R3 = max(R3, T9);"<<endl;
	output << "	"<<endl;
	output << "	R3.xyz = max(R3.xyz, R3.yzw);"<<endl;
	output << "	R3.yzw = max(R3.xyz, R3.yzw);"<<endl;
	output << "	"<<endl;
	output << "	R3.xyz = max(R3.xyz, T3.yzw);"<<endl;
	output << "	R3.yzw = max(T3.xyz, R3.yzw);	"<<endl;
	output << "	T7 = (T3> max(R3, threshold));"<<endl;
	output << "	"<<endl;
	output << "	float4 pixval;	"<<endl;
	output << "	pixval.x = dot(float4(0, 4, 8, 0), clamp(T5+T7, 0, 1));"<<endl;
	output << "	pixval.y = dot(float4(0, 4, 8, 0), clamp(T6+T7, 0, 1));"<<endl;
	output << "	pixval.z = dot(float4(0, 4, 8, 0), clamp(T4+T5+T6+T7, 0, 1));"<<endl;
	output << "	pixval.w = octave;"<<endl;
	output << ""<<endl;
	output << "	result = pixval / 255.0;"<<endl;
	output << "}"<<endl;
	output << ""<<endl;
	output.close();

//GPGPU_nms_other_fp.glsl
	output.open("GPGPU_nms_other_fp.glsl");
	output << displayCopyright();
	output << "void GPGPU_nms_other_fp("<<endl;
	output << "	float4 uv			   : TEXCOORD0,"<<endl;
	output << "	uniform sampler2D tex0 : register(s0),"<<endl;
	output << "	uniform float octave,"<<endl;
	output << "	uniform float threshold,"<<endl;
	output << "	out float4 result	   : COLOR"<<endl;
	output << ")"<<endl;
	output << "{"<<endl;
	output << displayStepXY_glsl()<<endl;
	output << "	"<<endl;
	output << "	//Octave o"<<endl;
	output << "	float stepX = steps[octave].x;"<<endl;
	output << "	float stepY = steps[octave].y;"<<endl;
	output << "	"<<endl;
	output << "	float4 coord = float4(uv.x-1.5*stepX, uv.y-1.5*stepY, 0, octave);"<<endl;
	output << "	"<<endl;
	output << "	float4 T0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, 0));"<<endl;
	output << "	float4 T2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, 0));"<<endl;
	output << "	float4 T3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, 0));"<<endl;
	output << ""<<endl;
	output << "	float4 T4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, 0)));"<<endl;
	output << "	float4 T5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, 0)));"<<endl;
	output << "	float4 T7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, 0)));"<<endl;
	output << "	"<<endl;
	output << "	float4 R0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, 0)), T3);"<<endl;
	output << "	float4 R1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, 0)), T2);"<<endl;
	output << "	float4 R2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, 0)), T1);"<<endl;
	output << "	float4 R3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, 0)), T0);"<<endl;
	output << ""<<endl;
	output << "	float4 T8 = max(T0, T3);"<<endl;
	output << "	float4 T9 = max(T1, T2);"<<endl;
	output << "	"<<endl;
	output << "	//R0"<<endl;
	output << "	R0 = max(R0, T4);"<<endl;
	output << "	R0 = max(R0, T5);"<<endl;
	output << "	R0 = max(R0, T9);"<<endl;
	output << "	"<<endl;
	output << "	//R1"<<endl;
	output << "	R1 = max(R1, T4);"<<endl;
	output << "	R1 = max(R1, T6);"<<endl;
	output << "	R1 = max(R1, T8);"<<endl;
	output << "	"<<endl;
	output << "	//R2"<<endl;
	output << "	R2 = max(R2, T5);"<<endl;
	output << "	R2 = max(R2, T7);"<<endl;
	output << "	R2 = max(R2, T8);"<<endl;
	output << "	"<<endl;
	output << "	//R3"<<endl;
	output << "	R3 = max(R3, T6);"<<endl;
	output << "	R3 = max(R3, T7);"<<endl;
	output << "	R3 = max(R3, T9);"<<endl;
	output << "	"<<endl;
	output << "	//Octave o-1"<<endl;
	output << "	float2 L0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, -1)).zw;"<<endl;
	output << "	float2 L1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, -1)).zw;"<<endl;
	output << "	float2 L2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, -1)).zw;"<<endl;
	output << "	float2 L3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, -1)).zw;"<<endl;
	output << ""<<endl;
	output << "	float2 L4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, -1)).zw, tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, -1)).zw);"<<endl;
	output << "	float2 L5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, -1)).zw);"<<endl;
	output << "	float2 L6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, -1)).zw);"<<endl;
	output << "	float2 L7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, -1)).zw);"<<endl;
	output << "	"<<endl;
	output << "	float2 K0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, -1)).zw, L3);"<<endl;
	output << "	float2 K1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, -1)).zw, L2);"<<endl;
	output << "	float2 K2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, -1)).zw, L1);"<<endl;
	output << "	float2 K3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, -1)).zw, L0);"<<endl;
	output << ""<<endl;
	output << "	float2 L8 = max(L0, L3);"<<endl;
	output << "	float2 L9 = max(L1, L2);	"<<endl;
	output << "	"<<endl;
	output << "	//K0"<<endl;
	output << "	K0 = max(K0, L4);"<<endl;
	output << "	K0 = max(K0, L5);"<<endl;
	output << "	K0 = max(K0, L9);"<<endl;
	output << "	"<<endl;
	output << "	//K1"<<endl;
	output << "	K1 = max(K1, L4);"<<endl;
	output << "	K1 = max(K1, L6);"<<endl;
	output << "	K1 = max(K1, L8);"<<endl;
	output << "	"<<endl;
	output << "	//K2"<<endl;
	output << "	K2 = max(K2, L5);"<<endl;
	output << "	K2 = max(K2, L7);"<<endl;
	output << "	K2 = max(K2, L8);"<<endl;
	output << "		"<<endl;
	output << "	//K3"<<endl;
	output << "	K3 = max(K3, L6);"<<endl;
	output << "	K3 = max(K3, L7);"<<endl;
	output << "	K3 = max(K3, L9);"<<endl;
	output << "	"<<endl;
	output << "	//maximum between octaves o and o-1"<<endl;
	output << "	"<<endl;
	output << "	//T0 - K0"<<endl;
	output << "	T4 = max(float4(K0.y,  R0.xyz), R0);"<<endl;
	output << "	T4 = max(float4(K0.xy, R0.xy),  T4);"<<endl;
	output << "	"<<endl;
	output << "	T4 = max(T4, T0);"<<endl;
	output << "	T4 = max(T4, float4(L0.xy, T0.xy));"<<endl;
	output << "	T4 = (float4(L0.y, T0.xyz)>=max(T4, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//T1 - K1"<<endl;
	output << "	T5 = max(float4(K1.y,  R1.xyz), R1);"<<endl;
	output << "	T5 = max(float4(K1.xy, R1.xy),  T5);"<<endl;
	output << "	"<<endl;
	output << "	T5 = max(T5, T1);"<<endl;
	output << "	T5 = max(T5, float4(L1.xy, T1.xy));"<<endl;
	output << "	T5 = (float4(L1.y, T1.xyz)>=max(T5, threshold));"<<endl;
	output << "		"<<endl;
	output << "	//T2 - K2"<<endl;
	output << "	T6 = max(float4(K2.y,  R2.xyz), R2);"<<endl;
	output << "	T6 = max(float4(K2.xy, R2.xy),  T6);"<<endl;
	output << "	"<<endl;
	output << "	T6 = max(T6, T2);"<<endl;
	output << "	T6 = max(T6, float4(L2.xy, T2.xy));"<<endl;
	output << "	T6 = (float4(L2.y, T2.xyz)>=max(T6, threshold));"<<endl;
	output << "	"<<endl;
	output << "	//T3 - K3"<<endl;
	output << "	T7 = max(float4(K3.y,  R3.xyz), R3);"<<endl;
	output << "	T7 = max(float4(K3.xy, R3.xy),  T7);"<<endl;
	output << "	"<<endl;
	output << "	T7 = max(T7, T3);"<<endl;
	output << "	T7 = max(T7, float4(L3.xy, T3.xy));"<<endl;
	output << "	T7 = (float4(L3.y, T3.xyz)>=max(T7, threshold));"<<endl;
	output << "	"<<endl;
	output << "	float4 pixval;	"<<endl;
	output << "	pixval.x = dot(float4(1, 2, 4, 8), clamp(T5+T7, 0, 1));"<<endl;
	output << "	pixval.y = dot(float4(1, 2, 4, 8), clamp(T6+T7, 0, 1));"<<endl;
	output << "	pixval.z = dot(float4(1, 2, 4, 8), clamp(T4+T5+T6+T7, 0, 1));"<<endl;
	output << "	pixval.w = octave;"<<endl;
	output << ""<<endl;
	output << "	result = pixval / 255.0;"<<endl;
	output << "}"<<endl;
	output.close();
}

std::string GPUSurfScriptGenerator::displayStepX_hlsl()
{
	stringstream output;
	output.precision(12);
	output << "	const float steps["<<mNbOctave<<"] = {"<<endl;
	for (int i=0; i<mNbOctave; i++)
		output << "		"<<std::setw(12) <<1.0/(mVideoWidth / (1<<i))<<",\t//"<<mVideoWidth / (1<<i)<<endl;
	output << "	};"<<endl;	

	return output.str();
}

std::string GPUSurfScriptGenerator::displayStepY_hlsl()
{
	stringstream output;
	output.precision(12);
	output << "	const float steps["<<mNbOctave<<"] = {"<<endl;
	for (int i=0; i<mNbOctave; i++)
		output << "		"<<std::setw(12) <<1.0/(mVideoHeight / (1<<i))<<",\t//"<<mVideoHeight / (1<<i)<<endl;
	output << "	};"<<endl;	

	return output.str();
}

std::string GPUSurfScriptGenerator::displayStepXY_hlsl()
{
	stringstream output;
	output.precision(12);
	output << "	const float2 steps["<<mNbOctave<<"] = {"<<endl;
	for (int i=0; i<mNbOctave; i++)
		output << "		float2("<<std::setw(12) <<1.0/(mVideoWidth / (1<<i))<<",\t"<<std::setw(12)<<1.0/(mVideoHeight / (1<<i))<<"),\t//"<<mVideoWidth / (1<<i)<<"x"<<mVideoHeight / (1<<i)<<endl;
	output << "	};"<<endl;	

	return output.str();
}

std::string GPUSurfScriptGenerator::displayStepXYHalf_hlsl()
{
	stringstream output;
	output.precision(12);
	output << "	const float2 steps["<<mNbOctave<<"] = {"<<endl;
	for (int i=1; i<mNbOctave+1; i++)
		output << "		float2("<<std::setw(12) <<1.0/(mVideoWidth / (1<<i))<<",\t"<<std::setw(12)<<1.0/(mVideoHeight / (1<<i))<<"),\t//"<<mVideoWidth / (1<<i)<<"x"<<mVideoHeight / (1<<i)<<endl;
	output << "	};"<<endl;	

	return output.str();
}

std::string GPUSurfScriptGenerator::displayGaussianKernel_hlsl()
{
	int scales = 4; //1 scale per channel -> 4 scales (RGBA)
	std::vector<GaussianKernel> kernels;

	double k = pow(4.0, 1.0/(scales));
	double w = 0.0;

	for (int s=1; s<scales+1; ++s)
	{	
		w += pow(k, s) - pow(k, s-1);
		kernels.push_back(GaussianKernel(w));
	}

	unsigned int maxKernelSize = kernels[kernels.size()-1].data.size();

	std::vector<std::vector<float>> gaussians;
	for (int s=0; s<scales; ++s)
	{
		std::vector<float> gaussian;
		GaussianKernel kernel = kernels[s];
		unsigned int paddingSize = (unsigned int )((maxKernelSize - kernel.data.size()) / 2.0);

		for (unsigned int i=0; i<paddingSize; ++i)
			gaussian.push_back(0.0);
		for (unsigned int i=0; i<kernel.data.size(); ++i)
			gaussian.push_back(kernel.data[i] / kernel.norm);
		for (unsigned int i=0; i<paddingSize; ++i)
			gaussian.push_back(0.0);
		gaussians.push_back(gaussian);
	}

	stringstream ss;
	ss.precision(15);
	ss.setf( ios::fixed, ios::floatfield );
	ss << "const float4 kernel["<<maxKernelSize<<"] = {"<<endl;
	for (unsigned int i=0; i<maxKernelSize; ++i)
	{
		ss << "		float4(" << std::setw(15) << gaussians[0][i] << ",\t"<< std::setw(15) << gaussians[1][i] <<",\t"<<std::setw(15) << gaussians[2][i] <<",\t"<<std::setw(15)<<gaussians[3][i]<<")";
		if (i != 18)
			ss <<",";
		ss <<endl;
	}
	ss << "	};";

	return ss.str();
}

std::string GPUSurfScriptGenerator::displayStepX_glsl()
{
	return displayStepX_hlsl();
}

std::string GPUSurfScriptGenerator::displayStepY_glsl()
{
	return displayStepY_hlsl();
}

std::string GPUSurfScriptGenerator::displayStepXY_glsl()
{
	return displayStepXY_hlsl();
}

std::string GPUSurfScriptGenerator::displayStepXYHalf_glsl()
{
	return displayStepXYHalf_hlsl();
}

std::string GPUSurfScriptGenerator::displayGaussianKernel_glsl()
{
	return displayGaussianKernel_hlsl();
}

std::string GPUSurfScriptGenerator::displayCopyright()
{
	std::stringstream output;

	output << "/*"<<std::endl;
	output << "	Copyright (c) 2010 ASTRE Henri (http://www.visual-experiments.com)"<<std::endl;
	output << ""<<std::endl;
	output << "	Permission is hereby granted, free of charge, to any person obtaining a copy"<<std::endl;
	output << "	of this software and associated documentation files (the \"Software\"), to deal"<<std::endl;
	output << "	in the Software without restriction, including without limitation the rights"<<std::endl;
	output << "	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell"<<std::endl;
	output << "	copies of the Software, and to permit persons to whom the Software is"<<std::endl;
	output << "	furnished to do so, subject to the following conditions:"<<std::endl;
	output << ""<<std::endl;
	output << "	The above copyright notice and this permission notice shall be included in"<<std::endl;
	output << "	all copies or substantial portions of the Software."<<std::endl;
	output << ""<<std::endl;
	output << "	THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR"<<std::endl;
	output << "	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,"<<std::endl;
	output << "	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE"<<std::endl;
	output << "	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER"<<std::endl;
	output << "	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,"<<std::endl;
	output << "	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN"<<std::endl;
	output << "	THE SOFTWARE."<<std::endl;
	output << "*/"<<std::endl;
	output <<std::endl;

	return output.str();
}