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

#version 150

uniform sampler2D tex0;
in vec2 gl_TexCoord[1];
out vec4 result;

uniform float octave;

void main()
{
	vec2 steps[5] = vec2[](
		vec2(   0.0015625,	0.00208333333333),	//640x480
		vec2(    0.003125,	0.00416666666667),	//320x240
		vec2(     0.00625,	0.00833333333333),	//160x120
		vec2(      0.0125,	0.0166666666667),	//80x60
		vec2(       0.025,	0.0333333333333)	//40x30
	);

	float stepX = steps[int(octave)].x;
	float stepY = steps[int(octave)].y;

	vec2 coord = gl_TexCoord[0];
	
	vec4 Lxx = -2*textureLod(tex0, coord, octave);
	vec4 Lyy = Lxx;
	vec4 Lxy;
	Lxx += textureLod(tex0, coord+vec2(-2.0*stepX,        0.0), octave) + textureLod(tex0, coord+vec2(2.0*stepX,       0.0), octave);
	Lyy += textureLod(tex0, coord+vec2(       0.0, -2.0*stepY), octave) + textureLod(tex0, coord+vec2(      0.0, 2.0*stepY), octave);
	Lxy  = textureLod(tex0, coord+vec2(    -stepX,     -stepY), octave) + textureLod(tex0, coord+vec2(    stepX,     stepY), octave);
	Lxy -= textureLod(tex0, coord+vec2(     stepX,     -stepY), octave) + textureLod(tex0, coord+vec2(   -stepX,     stepY), octave);
	Lxx /= 4;
	Lyy /= 4;
	Lxy /= 4;	

	result = Lxx*Lyy-Lxy*Lxy;
}
