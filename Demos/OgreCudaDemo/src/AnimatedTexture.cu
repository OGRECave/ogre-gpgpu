/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1415926536f

/* 
 * Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard.
 */
__global__ void cudaKernelTexture2D(unsigned char* surface, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned char* pixel;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;
	
    // get a pointer to the pixel at (x,y)
    pixel = (unsigned char*)(surface + y*pitch) + 4*x;

	// populate it
	float value_x = 0.5f + 0.5f*cos(t + 10.0f*( (2.0f*x)/width  - 1.0f ) );
	float value_y = 0.5f + 0.5f*cos(t + 10.0f*( (2.0f*y)/height - 1.0f ) );

	// Color : DirectX BGRA, OpenGL RGBA
	pixel[0] = 255*(0.5f + 0.5f*cos(t));                          // blue
	pixel[1] = 255*(0.5*pixel[1]/255.0 + 0.5*pow(value_y, 3.0f)); // green
	pixel[2] = 255*(0.5*pixel[0]/255.0 + 0.5*pow(value_x, 3.0f)); // red
	pixel[3] = 255;                                               // alpha	
}

extern "C" void cudaTextureUpdate(void* deviceTexture, int width, int height, float t)
{
    dim3 Db = dim3(16, 16); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);
	
	size_t pitch = width*4;
    cudaKernelTexture2D<<<Dg,Db>>>((unsigned char*)deviceTexture, width, height, pitch, t);
}