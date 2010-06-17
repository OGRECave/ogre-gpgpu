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
uniform sampler1D tex1;

in vec2 gl_TexCoord[1];
out vec4 result;

uniform float nbFeature;

void main()
{
	vec2 steps[5] = vec2[](
		vec2(   0.0015625,	0.00208333333333),	//640x480
		vec2(    0.003125,	0.00416666666667),	//320x240
		vec2(     0.00625,	0.00833333333333),	//160x120
		vec2(      0.0125,	0.0166666666667),	  //80x60
		vec2(       0.025,	0.0333333333333)	  //40x30
	);

  if (gl_TexCoord[0].x*4096.0 > nbFeature)
  {
    result = vec4(0.0, 0.0, 0.0, 0.0);
  }
  else
  {
    vec4 feature = texture(tex1, gl_TexCoord[0].x);
    //x      = feature.x
    //y      = feature.y
    //scale  = feature.z
    //octave = feature.w
    
    if (feature.w == 0.0)
    {
      result = vec4(0.0, 0.0, feature.z, feature.w);
    }
    else
    {
      float stepX = steps[int(feature.w)].x;
      float stepY = steps[int(feature.w)].y;

      vec2 coord = vec2(feature.x*stepX, feature.y*stepY);//, 0, feature.w);
      
      vec3 T0, T1, T2, T3, T4, T5, T6, T7, T8;
      
      if (feature.z == 0.0) //hard case : scale = a
      {  
        T0 = vec3(texture2D(tex0, coord+vec2(-stepX, -stepY), feature.w-1.0).zw, texture2D(tex0, coord+vec2(-stepX, -stepY), feature.w).x);
        T1 = vec3(texture2D(tex0, coord+vec2(   0.0, -stepY), feature.w-1.0).zw, texture2D(tex0, coord+vec2(   0.0, -stepY), feature.w).x);
        T2 = vec3(texture2D(tex0, coord+vec2( stepX, -stepY), feature.w-1.0).zw, texture2D(tex0, coord+vec2( stepX, -stepY), feature.w).x);
        T3 = vec3(texture2D(tex0, coord+vec2(-stepX,    0.0), feature.w-1.0).zw, texture2D(tex0, coord+vec2(-stepX,    0.0), feature.w).x);
        T4 = vec3(texture2D(tex0, coord+vec2(   0.0,    0.0), feature.w-1.0).zw, texture2D(tex0, coord+vec2(   0.0,    0.0), feature.w).x);
        T5 = vec3(texture2D(tex0, coord+vec2( stepX,    0.0), feature.w-1.0).zw, texture2D(tex0, coord+vec2( stepX,    0.0), feature.w).x);
        T6 = vec3(texture2D(tex0, coord+vec2(-stepX,  stepY), feature.w-1.0).zw, texture2D(tex0, coord+vec2(-stepX,  stepY), feature.w).x);
        T7 = vec3(texture2D(tex0, coord+vec2(   0.0,  stepY), feature.w-1.0).zw, texture2D(tex0, coord+vec2(   0.0,  stepY), feature.w).x);
        T8 = vec3(texture2D(tex0, coord+vec2( stepX,  stepY), feature.w-1.0).zw, texture2D(tex0, coord+vec2( stepX,  stepY), feature.w).x);
      }
      else if (feature.z == 1.0) //hard case : scale = r
      {  
        T0 = vec3(texture2D(tex0, coord+vec2(-stepX, -stepY), feature.w-1.0).w, texture2D(tex0, coord+vec2(-stepX, -stepY), feature.w).xy);
        T1 = vec3(texture2D(tex0, coord+vec2(   0.0, -stepY), feature.w-1.0).w, texture2D(tex0, coord+vec2(   0.0, -stepY), feature.w).xy);
        T2 = vec3(texture2D(tex0, coord+vec2( stepX, -stepY), feature.w-1.0).w, texture2D(tex0, coord+vec2( stepX, -stepY), feature.w).xy);
        T3 = vec3(texture2D(tex0, coord+vec2(-stepX,    0.0), feature.w-1.0).w, texture2D(tex0, coord+vec2(-stepX,    0.0), feature.w).xy);
        T4 = vec3(texture2D(tex0, coord+vec2(   0.0,    0.0), feature.w-1.0).w, texture2D(tex0, coord+vec2(   0.0,    0.0), feature.w).xy);
        T5 = vec3(texture2D(tex0, coord+vec2( stepX,    0.0), feature.w-1.0).w, texture2D(tex0, coord+vec2( stepX,    0.0), feature.w).xy);
        T6 = vec3(texture2D(tex0, coord+vec2(-stepX,  stepY), feature.w-1.0).w, texture2D(tex0, coord+vec2(-stepX,  stepY), feature.w).xy);
        T7 = vec3(texture2D(tex0, coord+vec2(   0.0,  stepY), feature.w-1.0).w, texture2D(tex0, coord+vec2(   0.0,  stepY), feature.w).xy);
        T8 = vec3(texture2D(tex0, coord+vec2( stepX,  stepY), feature.w-1.0).w, texture2D(tex0, coord+vec2( stepX,  stepY), feature.w).xy);
      }
      else if (feature.z == 2.0) //trivial case : scale = g
      {
        T0 = texture2D(tex0, coord+vec2(-stepX, -stepY), feature.w).xyz;
        T1 = texture2D(tex0, coord+vec2(   0.0, -stepY), feature.w).xyz;
        T2 = texture2D(tex0, coord+vec2( stepX, -stepY), feature.w).xyz;
        T3 = texture2D(tex0, coord+vec2(-stepX,    0.0), feature.w).xyz;
        T4 = texture2D(tex0, coord+vec2(   0.0,    0.0), feature.w).xyz;
        T5 = texture2D(tex0, coord+vec2( stepX,    0.0), feature.w).xyz;
        T6 = texture2D(tex0, coord+vec2(-stepX,  stepY), feature.w).xyz;
        T7 = texture2D(tex0, coord+vec2(   0.0,  stepY), feature.w).xyz;
        T8 = texture2D(tex0, coord+vec2( stepX,  stepY), feature.w).xyz;
      }
      else //if (feature.z == 3.0) //trivial case : scale = b
      {  
        T0 = texture2D(tex0, coord+vec2(-stepX, -stepY), feature.w).yzw;
        T1 = texture2D(tex0, coord+vec2(   0.0, -stepY), feature.w).yzw;
        T2 = texture2D(tex0, coord+vec2( stepX, -stepY), feature.w).yzw;
        T3 = texture2D(tex0, coord+vec2(-stepX,    0.0), feature.w).yzw;
        T4 = texture2D(tex0, coord+vec2(   0.0,    0.0), feature.w).yzw;
        T5 = texture2D(tex0, coord+vec2( stepX,    0.0), feature.w).yzw;
        T6 = texture2D(tex0, coord+vec2(-stepX,  stepY), feature.w).yzw;
        T7 = texture2D(tex0, coord+vec2(   0.0,  stepY), feature.w).yzw;
        T8 = texture2D(tex0, coord+vec2( stepX,  stepY), feature.w).yzw;	
      }

      //deriv3D -> G
      float dX = 0.5*T5.y - 0.5*T3.y;
      float dY = 0.5*T7.y - 0.5*T1.y;
      float dS = 0.5*T4.z - 0.5*T4.x;
      
      //hessian3D -> H
      float dXX = T5.y + T3.y - 2.0*T4.y;
      float dYY = T7.y + T1.y - 2.0*T4.y;
      float dSS = T4.z + T4.x - 2.0*T4.y;
      float dXY = 0.25*(T8.y - T6.y - T2.y + T0.y);
      float dXS = 0.25*(T5.z - T3.z - T5.x + T3.x);
      float dYS = 0.25*(T7.z - T1.z - T7.x + T1.x);
      
      mat3x3 H = mat3x3(
        dXX, dXY, dXS,
        dXY, dYY, dYS,
        dXS, dYS, dSS
      );
      
      vec3 G = vec3(
        dX, dY, dS
      );
      
      float detH = determinant(H);
      
      if (detH != 0.0) 
      {
        mat3x3 Hinv = mat3x3(
          H[1][1]*H[2][2] - H[1][2]*H[2][1],	H[0][2]*H[2][1] - H[0][1]*H[2][2],	H[0][1]*H[1][2] - H[0][2]*H[1][1],
          H[1][2]*H[2][0] - H[1][0]*H[2][2],	H[0][0]*H[2][2] - H[0][2]*H[2][0],	H[0][2]*H[1][0] - H[0][0]*H[1][2],
          H[1][0]*H[2][1] - H[1][1]*H[2][0],	H[0][1]*H[2][0] - H[0][0]*H[2][1],	H[0][0]*H[1][1] - H[0][1]*H[1][0]
        );
        //detH *= 255;
        vec3 tmp = -1.0/detH * (Hinv * G);
      //tmp[0] *= stepX;
      //tmp[1] *= stepY;
        
        if (abs(tmp[0])<0.5 && abs(tmp[1])<0.5)
          result = vec4(tmp[0], tmp[1], feature.z, feature.w);
        else
          result = vec4(tmp[0], tmp[1], tmp[2], 0.0); //255
      }
      else 
      {
        if (dX == 0.0 && T5.y == 0.0)
			result = vec4(feature.x, feature.y, feature.z, 256.0);
		else
			result = vec4(0.0, 0.0, 0.0, 0.0); 
      }    
    }
  }
}
