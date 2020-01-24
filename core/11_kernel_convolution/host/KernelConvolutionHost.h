#pragma once

#include "cudaTools.h"
#include "Grid.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class KernelConvolutionHost
    {
    public:
	KernelConvolutionHost(const Grid& grid, uchar* tabInput, uchar* tabOutput, int w, int h, int radius, float* tabGMKernel);
	virtual ~KernelConvolutionHost(void);
	void run();

    private:
	dim3 dg;
	dim3 db;
	uchar* tabOutput;
	uchar* tabInput;
	float* tabKernel;
	uchar* ptrDevGMInput;
	uchar* ptrDevGMOutput;
	int w;
	int h;
	int radius;
	float* ptrDevGMKernel;
	int sizeKernel;
	int sizeImg;
    };



/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
