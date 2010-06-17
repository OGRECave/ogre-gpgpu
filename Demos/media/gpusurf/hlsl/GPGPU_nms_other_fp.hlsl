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

void GPGPU_nms_other_fp(
	float4 uv			   : TEXCOORD0,
	uniform sampler2D tex0 : register(s0),
	uniform float octave,
	uniform float threshold,
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

	
	//Octave o
	float stepX = steps[octave].x;
	float stepY = steps[octave].y;
	
	float4 coord = float4(uv.x-1.5*stepX, uv.y-1.5*stepY, 0, octave);
	
	float4 T0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, 0));
	float4 T1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, 0));
	float4 T2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, 0));
	float4 T3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, 0));

	float4 T4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, 0)));
	float4 T5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, 0)));
	float4 T6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, 0)), tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, 0)));
	float4 T7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, 0)), tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, 0)));
	
	float4 R0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, 0)), T3);
	float4 R1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, 0)), T2);
	float4 R2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, 0)), T1);
	float4 R3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, 0)), T0);

	float4 T8 = max(T0, T3);
	float4 T9 = max(T1, T2);
	
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
	float2 L0 = tex2Dlod(tex0, coord+float4(  stepX,   stepY, 0, -1)).zw;
	float2 L1 = tex2Dlod(tex0, coord+float4(2*stepX,   stepY, 0, -1)).zw;
	float2 L2 = tex2Dlod(tex0, coord+float4(  stepX, 2*stepY, 0, -1)).zw;
	float2 L3 = tex2Dlod(tex0, coord+float4(2*stepX, 2*stepY, 0, -1)).zw;

	float2 L4 = max(tex2Dlod(tex0, coord+float4(  stepX,       0, 0, -1)).zw, tex2Dlod(tex0, coord+float4(2*stepX,       0, 0, -1)).zw);
	float2 L5 = max(tex2Dlod(tex0, coord+float4(      0,   stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(      0, 2*stepY, 0, -1)).zw);
	float2 L6 = max(tex2Dlod(tex0, coord+float4(3*stepX,   stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(3*stepX, 2*stepY, 0, -1)).zw);
	float2 L7 = max(tex2Dlod(tex0, coord+float4(  stepX, 3*stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(2*stepX, 3*stepY, 0, -1)).zw);
	
	float2 K0 = max(tex2Dlod(tex0, coord+float4(      0,       0, 0, -1)).zw, L3);
	float2 K1 = max(tex2Dlod(tex0, coord+float4(3*stepX,       0, 0, -1)).zw, L2);
	float2 K2 = max(tex2Dlod(tex0, coord+float4(      0, 3*stepY, 0, -1)).zw, L1);
	float2 K3 = max(tex2Dlod(tex0, coord+float4(3*stepX, 3*stepY, 0, -1)).zw, L0);

	float2 L8 = max(L0, L3);
	float2 L9 = max(L1, L2);	
	
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
	T4 = max(float4(K0.y,  R0.xyz), R0);
	T4 = max(float4(K0.xy, R0.xy),  T4);
	
	T4 = max(T4, T0);
	T4 = max(T4, float4(L0.xy, T0.xy));
	T4 = (float4(L0.y, T0.xyz)>=max(T4, threshold));
	
	//T1 - K1
	T5 = max(float4(K1.y,  R1.xyz), R1);
	T5 = max(float4(K1.xy, R1.xy),  T5);
	
	T5 = max(T5, T1);
	T5 = max(T5, float4(L1.xy, T1.xy));
	T5 = (float4(L1.y, T1.xyz)>=max(T5, threshold));
		
	//T2 - K2
	T6 = max(float4(K2.y,  R2.xyz), R2);
	T6 = max(float4(K2.xy, R2.xy),  T6);
	
	T6 = max(T6, T2);
	T6 = max(T6, float4(L2.xy, T2.xy));
	T6 = (float4(L2.y, T2.xyz)>=max(T6, threshold));
	
	//T3 - K3
	T7 = max(float4(K3.y,  R3.xyz), R3);
	T7 = max(float4(K3.xy, R3.xy),  T7);
	
	T7 = max(T7, T3);
	T7 = max(T7, float4(L3.xy, T3.xy));
	T7 = (float4(L3.y, T3.xyz)>=max(T7, threshold));
	
	float4 pixval;	
	pixval.x = dot(float4(1, 2, 4, 8), clamp(T5+T7, 0, 1));
	pixval.y = dot(float4(1, 2, 4, 8), clamp(T6+T7, 0, 1));
	pixval.z = dot(float4(1, 2, 4, 8), clamp(T4+T5+T6+T7, 0, 1));
	pixval.w = octave;

	result = pixval / 255.0;
}
