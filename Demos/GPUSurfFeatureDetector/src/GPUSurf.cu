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

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudpp.h>
#include <Feature.h>

//DirectX
#define BINDEX 0
#define GINDEX 1
#define RINDEX 2
#define AINDEX 3

//OpenGL
/*
#define RINDEX 0
#define GINDEX 1
#define BINDEX 2
#define AINDEX 3
*/
void saveToFile(float* buffer, float* buffer2, unsigned int size);
void saveToFile(Feature* buffer, unsigned int size);
void checkCUDAError(const char *msg);

__device__ float getNbMaximum(unsigned char* pixel)
{
	unsigned char extremum = pixel[BINDEX];
	unsigned char mask = 1;
	
	float nbMaximumFound = 0;
	for (unsigned int s=0; s<4; ++s)
	{
		if (extremum & mask)
			nbMaximumFound++;
		mask = mask<<1;
	}
	return nbMaximumFound;
}
__device__ float extractFeatureLocation(unsigned char* pixel, Feature* features, unsigned int start_index, float xpos, float ypos, int octave)
{
	unsigned char extremum = pixel[BINDEX];
	unsigned char mask = 1;
	
	unsigned int nbMaximumFound = 0;
	for (unsigned int s=0; s<4; ++s)
	{
		if (extremum & mask)
		{			
			float x = xpos*2 + ((pixel[RINDEX] & mask) != 0);
			float y = ypos*2 + ((pixel[GINDEX] & mask) != 0);
			Feature feat(x, y, s, octave);
			features[start_index + nbMaximumFound] = feat;
			nbMaximumFound++;
		}
		mask = mask<<1;
	}
	return nbMaximumFound;
}

__global__ void extractFeatureLocationPass1(float* out_counter, unsigned char* in_texture, size_t width, size_t height)
{
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;

	if (x < width)
	{
		float nbMaximumFound = 0;
		for (unsigned int i=0; i<height; ++i)
		{
			unsigned char* pixel = (unsigned char*)(in_texture + i*width*4 + 4*x);
			nbMaximumFound += getNbMaximum(pixel);
		}
		out_counter[x] = nbMaximumFound;
	}
}

__global__ void extractFeatureLocationPass2(Feature* out_features, unsigned char* in_texture, float* in_index_start, int index_start, int octave, size_t width, size_t height)
{
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;

	if (x < width)
	{
		float nbMaximumFound = in_index_start[x] + index_start;
		for (unsigned int i=0; i<height; ++i)
		{
			unsigned char* pixel = (unsigned char*)(in_texture + i*width*4 + 4*x);
			nbMaximumFound += extractFeatureLocation(pixel, out_features, nbMaximumFound, x, i, octave);
		}
	}
}

__global__ void copyCuda2Text1D(float* out_texture, Feature* in_features, int nbFeatureFound, int width)
{
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;

	if (x < nbFeatureFound)
	{
		float* pixel = (float*) (out_texture + x*4);
		Feature* feature = &in_features[x];
		pixel[BINDEX] = feature->x;	
		pixel[GINDEX] = feature->y;
		pixel[RINDEX] = feature->scale;
		pixel[AINDEX] = feature->octave;
	}
	else if (x < width)
	{
		float* pixel = (float*) (out_texture + x*4);
		pixel[BINDEX] = 0;
		pixel[GINDEX] = 0;
		pixel[RINDEX] = 0;
		pixel[AINDEX] = 0;
	}
}

__global__ void copyText1D2Cuda(Feature* out_features, float* in_texture, int nbFeatureFound, int width)
{
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;

	if (x < nbFeatureFound)
	{
		float* pixel = (float*) (in_texture + x*4);
		Feature* feature = &out_features[x];		
		/*
		feature->x       += pixel[BINDEX];
		feature->y       += pixel[GINDEX];
		*/
		feature->x       = pixel[BINDEX];
		feature->y       = pixel[GINDEX];
		feature->scale   = pixel[RINDEX];
		feature->octave  = pixel[AINDEX];		
	}
	else if (x < width)
	{
		Feature* feature = &out_features[x];
		feature->scale   = 0;
		feature->y       = 0;
		feature->x       = 0;	
		feature->octave  = 0;
	}
}

extern "C" void copyCuda2Tex1D(int width, int height, void* deviceTexture, Feature* deviceFeatures, unsigned int nbFeatureFound)
{
	dim3 block(16, 1, 1);
	dim3 grid(width / block.x, 1, 1);

	copyCuda2Text1D<<<grid, block, 0>>>((float*) deviceTexture, deviceFeatures, nbFeatureFound, width);
}

extern "C" void copyTex1D2Cuda(Feature* deviceFeatures, int width, int height, void* deviceTexture, unsigned int nbFeatureFound)
{
	dim3 block(16, 1, 1);
	dim3 grid(width / block.x, 1, 1);

	copyText1D2Cuda<<<grid, block, 0>>>(deviceFeatures, (float*) deviceTexture, nbFeatureFound, width);
}

extern "C" int extractFeatureLocationCuda(size_t width, size_t height, void* deviceTexture, 
							   CUDPPHandle& scanPlan,
							   int octave,
							   float* devicePass1, 
							   float* devicePass2,							   
#ifdef GPUSURF_HOST_DEBUG
							   float* hostPass1, 
							   float* hostPass2, 
#endif
							   Feature* deviceFeatures,
							   int featureStartIndex)
{
	dim3 block(16, 1, 1);
	dim3 grid(width / block.x, 1, 1);
	//printf("[%d] %dx%d -> %d\n", octave, width, height, grid.x);

	extractFeatureLocationPass1<<<grid, block, 0>>>(devicePass1, (unsigned char*) deviceTexture, width, height);

	cudppScan(scanPlan, devicePass2, devicePass1, width);

	extractFeatureLocationPass2<<<grid, block, 0>>>(deviceFeatures, (unsigned char*) deviceTexture, devicePass2, featureStartIndex, octave, width, height);	

	float nbFeature = 0;
	cudaMemcpy(&nbFeature, devicePass2+(width-1), sizeof(float), cudaMemcpyDeviceToHost);

#ifdef GPUSURF_HOST_DEBUG
	memset(hostPass1, 0, width*sizeof(float));
	memset(hostPass2, 0, width*sizeof(float));
	cudaMemcpy(hostPass1, devicePass1, width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostPass2, devicePass2, width*sizeof(float), cudaMemcpyDeviceToHost);
	
	//printf("[%d] nb feature found = %4f = %4f\n", octave, nbFeature, hostPass2[width-1]);

	saveToFile(hostPass1, hostPass2, width);
#endif

	return (int) nbFeature;
}

void saveToFile(float* buffer, float* buffer2, unsigned int size)
{
	FILE* fp = fopen("test.txt", "w");
	for (unsigned int i=0; i<size; ++i)
	{
		fprintf(fp, "[%d] -> %8.3f\t %8.3f\n", i, buffer[i], buffer2[i]);
	}
	fclose(fp);
}

void saveToFile(Feature* buffer, unsigned int size)
{
	FILE* fp = fopen("test_feature.txt", "w");
	for (unsigned int i=0; i<size; ++i)
	{
		fprintf(fp, "[%d] -> %8.3f\t %8.3f\n", i, buffer[i].x, buffer[i].y);
	}
	fclose(fp);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }                         
}