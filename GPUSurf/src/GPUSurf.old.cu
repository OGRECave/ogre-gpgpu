#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d9_interop.h>
#include <cudaD3D9.h>
#include <cudpp.h>
#include <Feature.h>

//http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/docs/online/modules.html

/*
	pixel[0] //B
	pixel[1] //G
	pixel[2] //R
	pixel[3] //A
*/

void saveToFile(float* buffer, float* buffer2, unsigned int size);
void saveToFile(Feature* buffer, unsigned int size);
void checkCUDAError(const char *msg);

__device__ float getNbMaximum(unsigned char* pixel)
{
	unsigned char extremum = pixel[0];
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
	unsigned char extremum = pixel[0];
	unsigned char mask = 1;
	
	unsigned int nbMaximumFound = 0;
	for (unsigned int s=0; s<4; ++s)
	{
		if (extremum & mask)
		{			
			float x = xpos*2 + ((pixel[2] & mask) != 0);
			float y = ypos*2 + ((pixel[1] & mask) != 0);
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
		pixel[0] = feature->x;	
		pixel[1] = feature->y;
		pixel[2] = feature->scale;
		pixel[3] = feature->octave;
	}
	else if (x < width)
	{
		float* pixel = (float*) (out_texture + x*4);
		pixel[0] = 0;
		pixel[1] = 0;
		pixel[2] = 0;
		pixel[3] = 0;
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
		feature->x       += pixel[0];
		feature->y       += pixel[1];
		*/
		feature->x       = pixel[0];
		feature->y       = pixel[1];
		feature->scale   = pixel[2];
		feature->octave  = pixel[3];		
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

void checkCUDAError(const char *msg);

extern "C" void registerTextureCuda(IDirect3DBaseTexture9* _texture)
{
	cuD3D9RegisterResource((LPDIRECT3DRESOURCE9)_texture, CU_D3D9_REGISTER_FLAGS_NONE);
}

extern "C" void unregisterTextureCuda(IDirect3DBaseTexture9* _texture)
{
	cuD3D9UnregisterResource((LPDIRECT3DRESOURCE9)_texture);
}

extern "C" void mapTextureCuda(IDirect3DBaseTexture9* texture)
{
	cudaD3D9MapResources(1, (IDirect3DResource9 **)&texture);
}

extern "C" void unmapTextureCuda(IDirect3DBaseTexture9* texture)
{
	cudaD3D9UnmapResources(1, (IDirect3DResource9 **)&texture);
}

extern "C" void copyCuda2Tex1D(IDirect3DBaseTexture9* texture, Feature* deviceFeatures, unsigned int nbFeatureFound)
{
	size_t width  = 0;
	size_t height = 0;
	void* deviceTexture;

	cudaD3D9ResourceGetSurfaceDimensions(&width, &height, NULL, texture, 0, 0);
	cudaD3D9ResourceGetMappedPointer(&deviceTexture, texture, 0, 0);

	dim3 block(16, 1, 1);
	dim3 grid(width / block.x, 1, 1);

	copyCuda2Text1D<<<grid, block, 0>>>((float*) deviceTexture, deviceFeatures, nbFeatureFound, width);
}

extern "C" void copyTex1D2Cuda(Feature* deviceFeatures, IDirect3DBaseTexture9* texture, unsigned int nbFeatureFound)
{
	size_t width  = 0;
	size_t height = 0;
	void* deviceTexture;

	cudaD3D9ResourceGetSurfaceDimensions(&width, &height, NULL, texture, 0, 0);
	cudaD3D9ResourceGetMappedPointer(&deviceTexture, texture, 0, 0);

	dim3 block(16, 1, 1);
	dim3 grid(width / block.x, 1, 1);

	copyText1D2Cuda<<<grid, block, 0>>>(deviceFeatures, (float*) deviceTexture, nbFeatureFound, width);
}

extern "C" int extractFeatureLocationCuda(IDirect3DBaseTexture9* texture, 
							   CUDPPHandle& scanPlan,
							   int octave,
							   float* devicePass1, 
							   float* devicePass2,							   
#ifdef GPUSURF_HOST_DEBUG
							   float* hostPass1, 
							   float* hostPass2, 
#endif
							   Feature* deviceFeatures,
							   Feature* hostFeatures,
							   int featureStartIndex)
{
	size_t width  = 0;
	size_t height = 0;
	void* deviceTexture;

	cudaD3D9ResourceGetSurfaceDimensions(&width, &height, NULL, texture, 0, octave);
	cudaD3D9ResourceGetMappedPointer(&deviceTexture, texture, 0, octave);
	//printf("%d %d %d\n", octave, width, height);

	dim3 block(16, 1, 1);
	dim3 grid(width / block.x, 1, 1);

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
	if (octave == 0)
		saveToFile(hostPass1, hostPass2, width);
#endif
	//cudaMemcpy(hostFeatures, deviceFeatures, width*sizeof(Feature), cudaMemcpyDeviceToHost);

	return (int) nbFeature;
}

void saveToFile(float* buffer, float* buffer2, unsigned int size)
{
	FILE* fp = fopen("test.txt", "w");
	for (unsigned int i=0; i<size; ++i)
	{
		fprintf(fp, "[%d] -> %8.3f\t %8.3f\n", i, buffer[i], buffer2[i]);
	}
	//fprintf(fp, "\n");
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