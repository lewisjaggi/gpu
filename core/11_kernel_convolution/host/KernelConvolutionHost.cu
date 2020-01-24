#include "KernelConvolutionHost.h"
#include <iostream>
#include <assert.h>
#include "Device.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/
extern __global__ void kernelConvolutionV1(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius, float* tabGMkernel);
extern __global__ void kernelConvolutionV2(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius, float* tabGMkernel);
extern __global__ void kernelConvolutionCM(uchar* tabGMInput, uchar* tabGMOutput, int w, int h, int radius);
extern __global__ void kernelConvolutionTexture(uchar* tabGMOutput, uint w, uint h, int kernelSize);

extern __host__ void uploadKernelConvolutionToCM(float* ptrKernelConvolution, int kernelSize);
extern __host__ void uploadImageAsTexture(uchar* ptrGMInput, uint w, uint h);
extern __host__ void unloadImageTexture();


/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*-------------------------------------*\
 |*		Constructeur		*|
 \*-------------------------------------*/

KernelConvolutionHost::KernelConvolutionHost(const Grid& grid, uchar* tabInput, uchar* tabOutput, int w, int h, int radius, float* tabKernel)
: w(w), h(h), radius(radius), tabInput(tabInput), tabOutput(tabOutput), tabKernel(tabKernel)
    {
    this->dg = grid.dg;
    this->db = grid.db;
    this->sizeKernel = (2*radius + 1)*(2*radius + 1)*sizeof(int);
    this->sizeImg = w*h*sizeof(uchar);

    Device::malloc(&ptrDevGMInput, sizeImg);
    Device::malloc(&ptrDevGMOutput, sizeImg);
    Device::malloc(&ptrDevGMKernel, sizeKernel);

    Device::memcpyHToD(ptrDevGMInput, tabInput, sizeImg);
    Device::memcpyHToD(ptrDevGMKernel, tabKernel, sizeKernel);
    Device::memclear(ptrDevGMOutput, sizeImg);

    uploadKernelConvolutionToCM(tabKernel, 2 * radius + 1);
    }

KernelConvolutionHost::~KernelConvolutionHost(void)
    {
    Device::free(ptrDevGMInput);
    Device::free(ptrDevGMKernel);
    Device::free(ptrDevGMOutput);
    }

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void KernelConvolutionHost::run()
    {
    //V2.0 CM
	{
//	kernelConvolutionCM<<<dg, db>>>(ptrDevGMInput, ptrDevGMOutput, w, h, radius);
	}

    //V3.0 Texture + CM
	{
	uploadImageAsTexture(ptrDevGMInput, (uint) w, (uint) h);
	kernelConvolutionTexture<<<dg,db>>>(ptrDevGMOutput, w, h, radius*radius+1);
	unloadImageTexture();
	}

    Device::memcpyDToH(tabOutput, ptrDevGMOutput, sizeImg);
    for (int i = 0; i < w * h; i++) //Affichage du résultat
	{
	if (i % w == 0)
	    {
	    std::cout << std::endl;
	    }
	std::cout << (int) tabOutput[i] << " ; ";
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

