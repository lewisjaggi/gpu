#pragma once

#include <iostream>
#include "cudaTools.h"
#include "MathTools.h"

#include "CVCaptureVideo.h"
#include "Animable_I_GPU.h"
#include "Version.h"
#include "FrameProvider.h"

using namespace gpu;

using cv::Mat;
using std::string;

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
	void openMPConvolution();
	void openMPMinMax();
	void amplificationOpenMP();

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:
	uint w;
	uint h;
	uint radius;
	int kernelSize;
	float* tabKernelConvolution;
	bool grayOnDevice;
	bool convolutionOnDevice;

	//Tools
	uchar4* tabGMImageCouleur;
	uchar* tabGMImageGris;

	uchar* tabGMMinMax;
	uchar* tabGMIntervalle; //Size == 2
	uchar* tabGMConvolutionOutput;
	float* tabGMKernelConvolution;

	size_t sizeImage;
	uchar4* ptrTabPixelVideo;
	uchar* ptrTabPixelGray;

	//Output

	uchar* tabImageConvolutionOutput;
	uchar* tabMinMaxOmp;

	Version version;

	FrameProvider frameProvider;

    };
