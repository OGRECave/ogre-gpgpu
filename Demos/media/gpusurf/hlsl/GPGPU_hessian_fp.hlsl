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

void GPGPU_hessian_fp(
	float4 uv			   : TEXCOORD0,
	uniform sampler2D tex0 : register(s0),
	uniform float octave,
	out float4 result	   : COLOR
)
{
	const float2 steps[5] = {
		float2(   0.0015625,	0.00208333333333),	//640x480
		float2(    0.003125,	0.00416666666667),	//320x240
		float2(     0.00625,	0.00833333333333),	//160x120
		float2(      0.0125,	0.0166666666667),	//80x60
		float2(       0.025,	0.0333333333333),	//40x30
	};

	const float stepX = steps[octave].x;
	const float stepY = steps[octave].y;

	float4 coord = float4(uv.x, uv.y, 0, octave);

	float4 Lxx = -2*tex2Dlod(tex0, coord);
	float4 Lyy = Lxx;
	float4 Lxy;
	Lxx += tex2Dlod(tex0, coord+float4(-2*stepX,0,0,0)) + tex2Dlod(tex0, coord+float4(2*stepX,0,0,0));
	Lyy += tex2Dlod(tex0, coord+float4(0,-2*stepY,0,0))    + tex2Dlod(tex0, coord+float4(0,2*stepY,0,0));
	Lxy  = tex2Dlod(tex0, coord+float4(-stepX,-stepY,0,0)) + tex2Dlod(tex0, coord+float4(stepX,stepY,0,0));
	Lxy -= tex2Dlod(tex0, coord+float4(stepX,-stepY,0,0))  + tex2Dlod(tex0, coord+float4(-stepX,stepY,0,0));
	Lxx /= 4;
	Lyy /= 4;
	Lxy /= 4;

	result = Lxx*Lyy-Lxy*Lxy;
}
