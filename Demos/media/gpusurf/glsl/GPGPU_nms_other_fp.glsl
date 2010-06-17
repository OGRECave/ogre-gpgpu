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
	
	//Octave o
	float stepX = steps[int(octave)].x;
	float stepY = steps[int(octave)].y;
	
	vec2 coord = vec2(gl_TexCoord[0].x-1.5*stepX, gl_TexCoord[0].y-1.5*stepY);
	
	vec4 T0 = textureLod(tex0, coord+vec2(    stepX,     stepY), octave);
	vec4 T1 = textureLod(tex0, coord+vec2(2.0*stepX,     stepY), octave);
	vec4 T2 = textureLod(tex0, coord+vec2(    stepX, 2.0*stepY), octave);
	vec4 T3 = textureLod(tex0, coord+vec2(2.0*stepX, 2.0*stepY), octave);

	vec4 T4 = max(textureLod(tex0, coord+vec2(    stepX,       0.0), octave), textureLod(tex0, coord+vec2(2.0*stepX,       0.0), octave));
	vec4 T5 = max(textureLod(tex0, coord+vec2(        0,     stepY), octave), textureLod(tex0, coord+vec2(      0.0, 2.0*stepY), octave));
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
	
	//R1
	R1 = max(R1, T4);
	R1 = max(R1, T6);
	R1 = max(R1, T8);
	
	//R2
	R2 = max(R2, T5);
	R2 = max(R2, T7);
	R2 = max(R2, T8);
	
	//R3
	R3 = max(R3, T6);
	R3 = max(R3, T7);
	R3 = max(R3, T9);
	
	//Octave o-1
	vec2 L0 = textureLod(tex0, coord+vec2(    stepX,     stepY), octave-1.0).zw;
	vec2 L1 = textureLod(tex0, coord+vec2(2.0*stepX,     stepY), octave-1.0).zw;
	vec2 L2 = textureLod(tex0, coord+vec2(    stepX, 2.0*stepY), octave-1.0).zw;
	vec2 L3 = textureLod(tex0, coord+vec2(2.0*stepX, 2.0*stepY), octave-1.0).zw;

	vec2 L4 = max(textureLod(tex0, coord+vec2(    stepX,       0.0), octave-1.0).zw, textureLod(tex0, coord+vec2(2.0*stepX,       0.0), octave-1.0).zw);
	vec2 L5 = max(textureLod(tex0, coord+vec2(      0.0,     stepY), octave-1.0).zw, textureLod(tex0, coord+vec2(      0.0, 2.0*stepY), octave-1.0).zw);
	vec2 L6 = max(textureLod(tex0, coord+vec2(3.0*stepX,     stepY), octave-1.0).zw, textureLod(tex0, coord+vec2(3.0*stepX, 2.0*stepY), octave-1.0).zw);
	vec2 L7 = max(textureLod(tex0, coord+vec2(    stepX, 3.0*stepY), octave-1.0).zw, textureLod(tex0, coord+vec2(2.0*stepX, 3.0*stepY), octave-1.0).zw);
	
	vec2 K0 = max(textureLod(tex0, coord+vec2(      0.0,       0.0), octave-1.0).zw, L3);
	vec2 K1 = max(textureLod(tex0, coord+vec2(3.0*stepX,       0.0), octave-1.0).zw, L2);
	vec2 K2 = max(textureLod(tex0, coord+vec2(      0.0, 3.0*stepY), octave-1.0).zw, L1);
	vec2 K3 = max(textureLod(tex0, coord+vec2(3.0*stepX, 3.0*stepY), octave-1.0).zw, L0);

	vec2 L8 = max(L0, L3);
	vec2 L9 = max(L1, L2);	
	
	//K0
	K0 = max(K0, L4);
	K0 = max(K0, L5);
	K0 = max(K0, L9);
	
	//K1
	K1 = max(K1, L4);
	K1 = max(K1, L6);
	K1 = max(K1, L8);
	
	//K2
	K2 = max(K2, L5);
	K2 = max(K2, L7);
	K2 = max(K2, L8);
		
	//K3
	K3 = max(K3, L6);
	K3 = max(K3, L7);
	K3 = max(K3, L9);
	
	//maximum between octaves o and o-1
	
	//T0 - K0
	T4 = max(vec4(K0.y,  R0.xyz), R0);
	T4 = max(vec4(K0.xy, R0.xy),  T4);
	
	T4 = max(T4, T0);
	T4 = max(T4, vec4(L0.xy, T0.xy));
	T4 = vec4(greaterThanEqual(vec4(L0.y, T0.xyz), max(T4, threshold))); //T4 = (vec4(L0.y, T0.xyz)>=max(T4, threshold));
	
	//T1 - K1
	T5 = max(vec4(K1.y,  R1.xyz), R1);
	T5 = max(vec4(K1.xy, R1.xy),  T5);
	
	T5 = max(T5, T1);
	T5 = max(T5, vec4(L1.xy, T1.xy));
	T5 = vec4(greaterThanEqual(vec4(L1.y, T1.xyz), max(T5, threshold))); //T5 = (vec4(L1.y, T1.xyz)>=max(T5, threshold));
		
	//T2 - K2
	T6 = max(vec4(K2.y,  R2.xyz), R2);
	T6 = max(vec4(K2.xy, R2.xy),  T6);
	
	T6 = max(T6, T2);
	T6 = max(T6, vec4(L2.xy, T2.xy));
	T6 = vec4(greaterThanEqual(vec4(L2.y, T2.xyz), max(T6, threshold))); //T6 = (vec4(L2.y, T2.xyz)>=max(T6, threshold));
	
	//T3 - K3
	T7 = max(vec4(K3.y,  R3.xyz), R3);
	T7 = max(vec4(K3.xy, R3.xy),  T7);
	
	T7 = max(T7, T3);
	T7 = max(T7, vec4(L3.xy, T3.xy));
	T7 = vec4(greaterThanEqual(vec4(L3.y, T3.xyz), max(T7, threshold))); //T7 = (vec4(L3.y, T3.xyz)>=max(T7, threshold));
	
	vec4 pixval;	
	pixval.x = dot(vec4(1.0, 2.0, 4.0, 8.0), clamp(T5+T7, 0.0, 1.0));
	pixval.y = dot(vec4(1.0, 2.0, 4.0, 8.0), clamp(T6+T7, 0.0, 1.0));
	pixval.z = dot(vec4(1.0, 2.0, 4.0, 8.0), clamp(T4+T5+T6+T7, 0.0, 1.0));
	pixval.w = octave;

	result = pixval / 255.0;
}
