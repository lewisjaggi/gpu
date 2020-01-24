#pragma once

#include <iostream>
#include "cudaTools.h"
#include "MathTools.h"

#include "CVCaptureVideo.h"
#include "Animable_I_GPU.h"

using namespace gpu;

using cv::Mat;
using std::string;

enum Version
    {
    BASIC,
    CM,
    TEXTURE,
    OMP,
    FULL_LOAD,
    PROD_CONS,
    };

class Convolution: public Animable_I<uchar>
    {
    public:
	Convolution(const Grid& grid, uint w, uint h, string videoName, float kernel[], int kernelSize, Version version);
	virtual ~Convolution(void);

	/*-------------------------*\
	|*   Override Animable_I   *|
	 \*------------------------*/

	/**
	 * Call periodicly by the api
	 */
	virtual void process(uchar* ptrDevPixels, uint w, uint h, const DomaineMath& domaineMath);

	/**
	 * Call periodicly by the api
	 */
	virtual void animationStep();

	void ptrTabPixelVideoToGray();
	void openMPConvolution(uchar* tabInput, uchar* tabOutput, int w, int h, int radius, float* tabKernel);

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:
	//Input
	string nameVideo;
	uint w;
	uint h;
	int kernelSize;
	float* tabKernelConvolution;
	bool grayOnDevice;

	//Tools
	uchar4* tabGMImageCouleur;
	uchar* tabGMImageGris;

	uchar* tabGMMinMax;
	uchar* tabGMIntervalle; //Size == 2
	uchar* tabGMConvolutionOutput;
	float* tabGMKernelConvolution;

	Mat matRGBA;
	size_t sizeImage;
	CVCaptureVideo capture;
	uchar4* ptrTabPixelVideo;
	uchar* ptrTabPixelGray;

	//Output
	uchar* tabImageOutput;

	Version version;
    };
