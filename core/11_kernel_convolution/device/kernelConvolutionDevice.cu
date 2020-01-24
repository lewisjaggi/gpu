#include <assert.h>
#include <stdio.h>

#include "Indice2D.h"
#include "cudaTools.h"
#include "Device.h"
#include <Texture.h>

#include "IndiceTools_GPU.h"
using namespace gpu;

#define KERNEL_CONVOLUTION_SIZE_MAX 9

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

__constant__ float KERNEL_CONVOLUTION_CM[KERNEL_CONVOLUTION_SIZE_MAX * KERNEL_CONVOLUTION_SIZE_MAX];

texture<uchar, 2, cudaReadModeElementType> textureRef;

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void kernelConvolutionV1(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius, float* tabGMkernel);
__global__ void kernelConvolutionV2(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius, float* tabGMkernel);
__global__ void kernelConvolutionCM(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius);
__global__ void kernelConvolutionTexture(uchar* tabGMOutput, uint w, uint h, int kernelSize);

__host__ void uploadImageAsTexture(uchar* tabGMImage, uint w, uint h);
__host__ void unloadImageTexture();

__host__ void uploadKernelConvolutionToCM(float* ptrKernelConvolution, int kernelSize);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static __device__ void reductionIntraThreadConvolution(uchar* tabGMOutput, uint w, uint h, int kernel_size, float* tabCMKernel);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * must be called by host
 * ptrTabSpheres est un tableau de sphere cote host
 */
__host__ void uploadKernelConvolutionToCM(float* ptrKernelConvolution, int kernelSize)
    {
    assert(kernelSize <= KERNEL_CONVOLUTION_SIZE_MAX);
    int offset = 0;
    HANDLE_ERROR(cudaMemcpyToSymbol(KERNEL_CONVOLUTION_CM, ptrKernelConvolution, sizeof(float) * kernelSize * kernelSize, offset, cudaMemcpyHostToDevice));
    }

__host__ void uploadImageAsTexture(uchar* tabGMImage, uint w, uint h)
    {
    //Configuration texture, valeurs par defaut !
    textureRef.addressMode[0] = cudaAddressModeClamp;    //par defaut
    textureRef.addressMode[1] = cudaAddressModeClamp;    //par defaut
    textureRef.filterMode = cudaFilterModePoint;    //par defaut
    textureRef.normalized = false;    //coordonnée texture //par defaut
    size_t pitch = w * sizeof(uchar);    //size ligne
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();

    HANDLE_ERROR(cudaBindTexture2D(NULL, textureRef, tabGMImage, channelDesc, w, h, pitch));
    }

__host__ void unloadImageTexture()
    {
    HANDLE_ERROR(cudaUnbindTexture(textureRef));
    }

__global__ void kernelConvolutionTexture(uchar* tabGMOutput, uint w, uint h, int kernelSize)
    {
    reductionIntraThreadConvolution(tabGMOutput, w, h, kernelSize, KERNEL_CONVOLUTION_CM);
    }

__global__ void kernelConvolutionV1(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius, float* tabGMkernel)
    {
	int s = Indice2D::tid();
	while(s < w*h)
	{
	    int i = 0;
	    int j = 0;
	    int u = 0;
	    int v = 0;
	    IndiceTools::toIJ(s, w, &u, &v);
	    if(u < radius || v < radius || u >= h-radius || v >= w-radius)
		{
		return;
		}
	    float sum = 0;
	    int sizeLine = 2*radius + 1;
	    while(i < sizeLine)
		{
		int x = (v-radius+i);
		while(j < sizeLine)
		    {
		    int y = u-radius+j;
		    sum += tabGMkernel[j * sizeLine + i] * tabGMInput[w * y + x];
		    ++j;
		    }
		++i;
		j = 0;
		}
	    if(sum<0)
		{
		sum=0;
		}
	    else if(sum > 255)
		{
		sum = 255;
		}

	    tabGMOutput[s] =(int) sum;
	    s+=Indice2D::nbThreadLocal();
	}
    }

__global__ void kernelConvolutionV2(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius, float* tabGMkernel)
    {
	int s = Indice2D::tid();
	while(s < w*h)
	{
	    int i = 0;
	    int j = 0;
	    int u = 0;
	    int v = 0;
	    IndiceTools::toIJ(s, w, &u, &v);

	    float sum = 0;
	    int sizeLine = 2*radius + 1;
	    while(i < sizeLine)
		{
		int x = (v-radius+i);
		while(j < sizeLine)
		    {
		    int y = u-radius+j;
		    if(!(x < 0 || y < 0 || x >= w || y >= h))
			{
			sum += tabGMkernel[j * sizeLine + i] * tabGMInput[w * y + x];
			}
		    ++j;
		    }
		++i;
		j = 0;
		}
	    if(sum<0)
		{
		sum=0;
		}
	    else if(sum > 255)
		{
		sum = 255;
		}
	    tabGMOutput[s] =(int) sum;
	    s+=Indice2D::nbThreadLocal();
	}
    }

__global__ void kernelConvolutionCM(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius)
    {
	int s = Indice2D::tid();
	while(s < w*h)
	{
	    int i = 0;
	    int j = 0;
	    int u = 0;
	    int v = 0;
	    IndiceTools::toIJ(s, w, &u, &v);

	    float sum = 0;
	    int sizeLine = 2*radius + 1;
	    while(i < sizeLine)
		{
		int x = (v-radius+i);
		while(j < sizeLine)
		    {
		    int y = u-radius+j;
		    if(!(x < 0 || y < 0 || x >= w || y >= h))
			{
			sum += KERNEL_CONVOLUTION_CM[j * sizeLine + i] * tabGMInput[w * y + x];
			}
		    ++j;
		    }
		++i;
		j = 0;
		}
	    if(sum<0)
		{
		sum=0;
		}
	    else if(sum > 255)
		{
		sum = 255;
		}
	    tabGMOutput[s] =(int) sum;
	    s+=Indice2D::nbThreadLocal();
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__device__ void reductionIntraThreadConvolution(uchar* tabGMOutput, uint w, uint h, int kernel_size, float* tabCMKernel)
    {
    const int NB_THREADS = Indice2D::nbThreadLocal();

    const int TID = Indice2D::tidLocal();
    const int limit = w * h;

    int s = TID; // Pixel to manage
    const int k = kernel_size;
    const int ss = k*(k/2.);// Center of kernel

    while (s < limit)
	{
	float sum = 0;

	const int j = s % w;
	const int i = s / w;

	for(int v=1;v<=k/2;++v)
	    {
	    for(int u=1;u<=k/2;++u)
		{
		sum += tabCMKernel[ss + v*k + u] * (int)tex2D(textureRef, j + u, i+v);
		sum += tabCMKernel[ss + v*k - u] * (int)tex2D(textureRef, j - u, i+v);
		sum += tabCMKernel[ss - v*k + u] * (int)tex2D(textureRef, j + u, i-v);
		sum += tabCMKernel[ss - v*k - u] * (int)tex2D(textureRef, j - u, i-v);
		}
	    sum += tabCMKernel[ss + v] * (int)tex2D(textureRef, j+v, i);
	    sum += tabCMKernel[ss - v] * (int)tex2D(textureRef, j-v, i);

	    sum += tabCMKernel[ss + v*k] * (int)tex2D(textureRef, j, i+v);
	    sum += tabCMKernel[ss - v*k] * (int)tex2D(textureRef, j, i-v);
	    }

	sum += tabCMKernel[ss] * (int)tex2D(textureRef, j, i);

	if(sum < 0)
	    sum = 0;
	if(sum > 255)
	    sum = 255;

	tabGMOutput[s] = sum;

	// Update pixel s image raw major linéariser
	s += NB_THREADS;
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

