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
uniform float threshold;

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
	vec2 coord  = vec2(gl_TexCoord[0].x-1.5*stepX, gl_TexCoord[0].y-1.5*stepY);
	
	vec4 T0 = textureLod(tex0, coord+vec2(    stepX,     stepY), octave);
	vec4 T1 = textureLod(tex0, coord+vec2(2.0*stepX,     stepY), octave);
	vec4 T2 = textureLod(tex0, coord+vec2(    stepX, 2.0*stepY), octave);
	vec4 T3 = textureLod(tex0, coord+vec2(2.0*stepX, 2.0*stepY), octave);

	vec4 T4 = max(textureLod(tex0, coord+vec2(    stepX,       0.0), octave), textureLod(tex0, coord+vec2(2.0*stepX,       0.0), octave));
	vec4 T5 = max(textureLod(tex0, coord+vec2(      0.0,     stepY), octave), textureLod(tex0, coord+vec2(      0.0, 2.0*stepY), octave));
	vec4 T6 = max(textureLod(tex0, coord+vec2(3.0*stepX,     stepY), octave), textureLod(tex0, coord+vec2(3.0*stepX, 2.0*stepY), octave));
	vec4 T7 = max(textureLod(tex0, coord+vec2(    stepX, 3.0*stepY), octave), textureLod(tex0, coord+vec2(2.0*stepX, 3.0*stepY), octave));
	
	vec4 R0 = max(textureLod(tex0, coord+vec2(      0.0,       0.0), octave), T3);
	vec4 R1 = max(textureLod(tex0, coord+vec2(3.0*stepX,       0.0), octave), T2);
	vec4 R2 = max(textureLod(tex0, coord+vec2(      0.0, 3.0*stepY), octave), T1);
	vec4 R3 = max(textureLod(tex0, coord+vec2(3.0*stepX, 3.0*stepY), octave), T0);

	vec4 T8 = max(T0, T3);
	vec4 T9 = max(T1, T2);
	
	//R0
	R0 = max(R0, T4);
	R0 = max(R0, T5);
	R0 = max(R0, T9);
				
	R0.xyz = max(R0.xyz, R0.yzw);
	R0.yzw = max(R0.xyz, R0.yzw);
	
	R0.xyz = max(R0.xyz, T0.yzw);
	R0.yzw = max(T0.xyz, R0.yzw); 
	T4 = vec4(greaterThanEqual(T0, max(R0, threshold))); //T4 = (T0>=max(R0, threshold));
	
	//R1
	R1 = max(R1, T4);
	R1 = max(R1, T6);
	R1 = max(R1, T8);
	
	R1.xyz = max(R1.xyz, R1.yzw);
	R1.yzw = max(R1.xyz, R1.yzw);

	R1.xyz = max(R1.xyz, T1.yzw);
	R1.yzw = max(T1.xyz, R1.yzw);	
	T5 = vec4(greaterThan(T1, max(R1, threshold))); //T5 = (T1> max(R1, threshold));
	
	//R2
	R2 = max(R2, T5);
	R2 = max(R2, T7);
	R2 = max(R2, T8);
	
	R2.xyz = max(R2.xyz, R2.yzw);
	R2.yzw = max(R2.xyz, R2.yzw);
	
	R2.xyz = max(R2.xyz, T2.yzw);
	R2.yzw = max(T2.xyz, R2.yzw);	
	T6 = vec4(greaterThan(T2, max(R2, threshold))); //T6 = (T2> max(R2, threshold));
	
	//R3
	R3 = max(R3, T6);
	R3 = max(R3, T7);
	R3 = max(R3, T9);
	
	R3.xyz = max(R3.xyz, R3.yzw);
	R3.yzw = max(R3.xyz, R3.yzw);
	
	R3.xyz = max(R3.xyz, T3.yzw);
	R3.yzw = max(T3.xyz, R3.yzw);	
	T7 = vec4(greaterThan(T3, max(R3, threshold))); //T7 = (T3> max(R3, threshold));
	
	vec4 pixval;	
	pixval.x = dot(vec4(0.0, 4.0, 8.0, 0.0), clamp(T5+T7, 0.0, 1.0));
	pixval.y = dot(vec4(0.0, 4.0, 8.0, 0.0), clamp(T6+T7, 0.0, 1.0));
	pixval.z = dot(vec4(0.0, 4.0, 8.0, 0.0), clamp(T4+T5+T6+T7, 0.0, 1.0));
	pixval.w = octave;

	result = pixval / 255.0;
}
