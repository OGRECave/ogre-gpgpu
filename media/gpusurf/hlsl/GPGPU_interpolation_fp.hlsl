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

void GPGPU_interpolation_fp(
	float4 uv			   : TEXCOORD0,
	uniform sampler2D tex0 : register(s0),
	uniform sampler1D tex1 : register(s1),
	uniform float nbFeature,
	out float4 result	   : COLOR
)
{
	const float2 steps[5] = {
		float2(   0.0015625,	0.00208333333333),	//640x480
		float2(    0.003125,	0.00416666666667),	//320x240
		float2(     0.00625,	0.00833333333333),	//160x120
		float2(      0.0125,	0.0166666666667),	  //80x60
		float2(       0.025,	0.0333333333333),	  //40x30
	};

  if (uv.x*4096 > nbFeature)
  {
    result = float4(0, 0, 0, 0);
  }
  else
  {
    float4 feature = tex1D(tex1, uv.x);
    //x      = feature.x
    //y      = feature.y
    //scale  = feature.z
    //octave = feature.w
    
    if (feature.w == 0)
    {
      result = float4(0, 0, feature.z, feature.w);
    }
    else
    {
      float stepX = steps[feature.w].x;
      float stepY = steps[feature.w].y;

      float4 coord = float4(feature.x*stepX, feature.y*stepY, 0, feature.w);
      
      float3 T0, T1, T2, T3, T4, T5, T6, T7, T8;
      
      if (feature.z == 0) //hard case : scale = a
      {  
        T0 = float3(tex2Dlod(tex0, coord+float4(-stepX, -stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(-stepX, -stepY, 0, 0)).x);
        T1 = float3(tex2Dlod(tex0, coord+float4(     0, -stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(     0, -stepY, 0, 0)).x);
        T2 = float3(tex2Dlod(tex0, coord+float4( stepX, -stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4( stepX, -stepY, 0, 0)).x);
        T3 = float3(tex2Dlod(tex0, coord+float4(-stepX,      0, 0, -1)).zw, tex2Dlod(tex0, coord+float4(-stepX,      0, 0, 0)).x);
        T4 = float3(tex2Dlod(tex0, coord+float4(     0,      0, 0, -1)).zw, tex2Dlod(tex0, coord+float4(     0,      0, 0, 0)).x);
        T5 = float3(tex2Dlod(tex0, coord+float4( stepX,      0, 0, -1)).zw, tex2Dlod(tex0, coord+float4( stepX,      0, 0, 0)).x);
        T6 = float3(tex2Dlod(tex0, coord+float4(-stepX,  stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(-stepX,  stepY, 0, 0)).x);
        T7 = float3(tex2Dlod(tex0, coord+float4(     0,  stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4(     0,  stepY, 0, 0)).x);
        T8 = float3(tex2Dlod(tex0, coord+float4( stepX,  stepY, 0, -1)).zw, tex2Dlod(tex0, coord+float4( stepX,  stepY, 0, 0)).x);
      }
      else if (feature.z == 1) //hard case : scale = r
      {  
        T0 = float3(tex2Dlod(tex0, coord+float4(-stepX, -stepY, 0, -1)).w, tex2Dlod(tex0, coord+float4(-stepX, -stepY, 0, 0)).xy);
        T1 = float3(tex2Dlod(tex0, coord+float4(     0, -stepY, 0, -1)).w, tex2Dlod(tex0, coord+float4(     0, -stepY, 0, 0)).xy);
        T2 = float3(tex2Dlod(tex0, coord+float4( stepX, -stepY, 0, -1)).w, tex2Dlod(tex0, coord+float4( stepX, -stepY, 0, 0)).xy);
        T3 = float3(tex2Dlod(tex0, coord+float4(-stepX,      0, 0, -1)).w, tex2Dlod(tex0, coord+float4(-stepX,      0, 0, 0)).xy);
        T4 = float3(tex2Dlod(tex0, coord+float4(     0,      0, 0, -1)).w, tex2Dlod(tex0, coord+float4(     0,      0, 0, 0)).xy);
        T5 = float3(tex2Dlod(tex0, coord+float4( stepX,      0, 0, -1)).w, tex2Dlod(tex0, coord+float4( stepX,      0, 0, 0)).xy);
        T6 = float3(tex2Dlod(tex0, coord+float4(-stepX,  stepY, 0, -1)).w, tex2Dlod(tex0, coord+float4(-stepX,  stepY, 0, 0)).xy);
        T7 = float3(tex2Dlod(tex0, coord+float4(     0,  stepY, 0, -1)).w, tex2Dlod(tex0, coord+float4(     0,  stepY, 0, 0)).xy);
        T8 = float3(tex2Dlod(tex0, coord+float4( stepX,  stepY, 0, -1)).w, tex2Dlod(tex0, coord+float4( stepX,  stepY, 0, 0)).xy);
      }
      else if (feature.z == 2) //trivial case : scale = g
      {
        T0 = tex2Dlod(tex0, coord+float4(-stepX, -stepY, 0, 0)).xyz;
        T1 = tex2Dlod(tex0, coord+float4(     0, -stepY, 0, 0)).xyz;
        T2 = tex2Dlod(tex0, coord+float4( stepX, -stepY, 0, 0)).xyz;
        T3 = tex2Dlod(tex0, coord+float4(-stepX,      0, 0, 0)).xyz;
        T4 = tex2Dlod(tex0, coord+float4(     0,      0, 0, 0)).xyz;
        T5 = tex2Dlod(tex0, coord+float4( stepX,      0, 0, 0)).xyz;
        T6 = tex2Dlod(tex0, coord+float4(-stepX,  stepY, 0, 0)).xyz;
        T7 = tex2Dlod(tex0, coord+float4(     0,  stepY, 0, 0)).xyz;
        T8 = tex2Dlod(tex0, coord+float4( stepX,  stepY, 0, 0)).xyz;
      }
      else //if (feature.z == 3) //trivial case : scale = b
      {  
        T0 = tex2Dlod(tex0, coord+float4(-stepX, -stepY, 0, 0)).yzw;
        T1 = tex2Dlod(tex0, coord+float4(     0, -stepY, 0, 0)).yzw;
        T2 = tex2Dlod(tex0, coord+float4( stepX, -stepY, 0, 0)).yzw;
        T3 = tex2Dlod(tex0, coord+float4(-stepX,      0, 0, 0)).yzw;
        T4 = tex2Dlod(tex0, coord+float4(     0,      0, 0, 0)).yzw;
        T5 = tex2Dlod(tex0, coord+float4( stepX,      0, 0, 0)).yzw;
        T6 = tex2Dlod(tex0, coord+float4(-stepX,  stepY, 0, 0)).yzw;
        T7 = tex2Dlod(tex0, coord+float4(     0,  stepY, 0, 0)).yzw;
        T8 = tex2Dlod(tex0, coord+float4( stepX,  stepY, 0, 0)).yzw;	
      }

      //deriv3D -> G
      float dX = 0.5*T5.y - 0.5*T3.y;
      float dY = 0.5*T7.y - 0.5*T1.y;
      float dS = 0.5*T4.z - 0.5*T4.x;
      
      //hessian3D -> H
      float dXX = T5.y + T3.y - 2*T4.y;
      float dYY = T7.y + T1.y - 2*T4.y;
      float dSS = T4.z + T4.x - 2*T4.y;
      float dXY = 0.25*(T8.y - T6.y - T2.y + T0.y);
      float dXS = 0.25*(T5.z - T3.z - T5.x + T3.x);
      float dYS = 0.25*(T7.z - T1.z - T7.x + T1.x);
      
      float3x3 H = {
        dXX, dXY, dXS,
        dXY, dYY, dYS,
        dXS, dYS, dSS
      };
      
      float3x1 G = {
        dX, dY, dS
      };
      
      float detH = determinant(H);
      
      if (detH != 0.0) 
      {
        float3x3 Hinv = {
          H[1][1]*H[2][2] - H[1][2]*H[2][1],	H[0][2]*H[2][1] - H[0][1]*H[2][2],	H[0][1]*H[1][2] - H[0][2]*H[1][1],
          H[1][2]*H[2][0] - H[1][0]*H[2][2],	H[0][0]*H[2][2] - H[0][2]*H[2][0],	H[0][2]*H[1][0] - H[0][0]*H[1][2],
          H[1][0]*H[2][1] - H[1][1]*H[2][0],	H[0][1]*H[2][0] - H[0][0]*H[2][1],	H[0][0]*H[1][1] - H[0][1]*H[1][0]
        };
        //detH *= 255;
        float3x1 tmp = -1.0/detH*mul(Hinv, G);
      //tmp[0] *= stepX;
      //tmp[1] *= stepY;
        
        if (abs(tmp[0])<0.5 && abs(tmp[1])<0.5)
          result = float4(tmp[0], tmp[1], feature.z, feature.w);
        else
          result = float4(tmp[0], tmp[1], tmp[2], 0); //255
      }
      else 
      {
        if (dX == 0 && T5.y == 0)
			result = float4(feature.x, feature.y, feature.z, 256);
		else
			result = float4(0, 0, 0, 0); 
      }    
    }
  }
}
