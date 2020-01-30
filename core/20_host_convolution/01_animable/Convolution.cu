#include "Convolution.h"

#include <assert.h>
#include <iostream>
#include <boost/lockfree/spsc_queue.hpp>
#include "KernelGrisMath.h"

#include "OpencvTools_GPU.h"
#include "IndiceTools_CPU.h"
#include "OmpTools.h"
#include "Device.h"
#include "Interval_CPU.h"
#include "Calibreur_CPU.h"

using std::cout;
using std::cerr;
using std::endl;

/** Kernels **/
extern __global__ void kernelGris(uchar4* tabGMImageCouleur, uchar* tabGMImageGris, uint w, uint h);

__global__ void kernelConvolutionV1(uchar *tabGMInput, uchar *tabGMOutput, int w, int h, int radius, float *tabGMkernel);
__global__ void kernelConvolutionV2(uchar *tabGMInput, uchar *tabGMOutput, int w, int h, int radius, float *tabGMkernel);
__global__ void kernelConvolutionCM(uchar *tabGMInput, uchar *tabGMOutput, int w, int h, int radius);
__global__ void kernelConvolutionTexture(uchar *tabGMOutput, uint w, uint h, int kernelSize);

extern __global__ void kernelMinMax(uchar* tabGMConvolutionOutput, uchar* tabGMMinMax, int w , int h);
extern __global__ void kernelAmplification(uchar* tabGMConvolutionOutput, uchar* tabGMIntervalle, int w, int h);

extern __host__ void uploadKernelConvolutionToCM(float* ptrKernelConvolution, int kernelSize);
extern __host__ void uploadImageAsTexture(uchar* ptrGMInput, uint w, uint h);
extern __host__ void unloadImageTexture();

//TODO Choose where to use NoyauCreator, passé depuis le provider ou directement dans cette class!
Convolution::Convolution(const Grid &grid, uint w, uint h, string nameVideo, float kernel[], int kernelSize, Version version) :
	Animable_I<uchar>(grid, w, h, "Convolution_GRAY_uchar"), tabKernelConvolution(kernel), kernelSize(
		kernelSize), w(w), h(h), radius(kernelSize / 2), version(version), frameProvider(w, h, nameVideo, version)

    {
    if (version == Version::PROD_CONS)
	{
	frameProvider.start();
	}
    assert(kernelSize % 2 == 1);

    //Paramètres
    this->onDevice = true;
    this->kernelSize = 9;

    int nbPixels = w * h;
    int nbPixelConvolution = kernelSize * kernelSize;

    Device::malloc(&tabGMMinMax, 2 * sizeof(uchar));
    Device::malloc(&tabGMImageGris, nbPixels * sizeof(uchar));
    Device::malloc(&tabGMImageCouleur, nbPixels * sizeof(uchar4));
    Device::malloc(&tabGMConvolutionOutput, nbPixels * sizeof(uchar));
    Device::malloc(&tabGMKernelConvolution, nbPixelConvolution * sizeof(float));

    // load kernelConvolution
    Device::memcpyHToD(tabGMKernelConvolution, tabKernelConvolution, nbPixelConvolution * sizeof(float));
    uploadKernelConvolutionToCM(tabKernelConvolution, kernelSize);
    uploadImageAsTexture(tabGMImageGris, w, h);

    // Tools
    this->t = 0; // protected dans Animable
    this->sizeImage = sizeof(uchar4) * w * h;

    ptrTabPixelGray = new uchar[sizeImage];
    tabImageConvolutionOutput = new uchar[sizeImage];
    tabMinMaxOmp = new uchar[2];

    //video
    this->ptrTabPixelVideo = frameProvider.getFrame();
    }

Convolution::~Convolution()
    {
    //Free
    Device::free(&tabGMMinMax);
    Device::free(&tabGMImageGris);
    Device::free(&tabGMImageCouleur);
    Device::free(&tabGMConvolutionOutput);
    Device::free(&tabGMKernelConvolution);
    unloadImageTexture();
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

void Convolution::process(uchar *ptrDevPixels, uint w, uint h, const DomaineMath &domaineMath)
    {
    //Gris
    if (onDevice)
	{
	 dim3 dg = dim3(48, 1, 1);
	 dim3 db = dim3(704, 1, 1);
	Device::memcpyHToD(tabGMImageCouleur, ptrTabPixelVideo, sizeImage);
    kernelGris<<<dg,db>>>(tabGMImageCouleur, tabGMImageGris, w, h);

    if (version == Version::BASIC)
            {
            dim3 dg = dim3(14, 1, 1);
            dim3 db = dim3(1024, 1, 1);
            kernelConvolutionV2<<<dg,db>>>(tabGMImageGris, tabGMConvolutionOutput, w, h, 4, tabGMKernelConvolution);
            //kernelConvolutionCM<<<dg,db>>>(tabGMImageGris, tabGMConvolutionOutput, w, h, kernelSize/2);
            }

        //Convolution CM
        else if (version == Version::CM || version == Version::FULL_LOAD || version == Version::PROD_CONS)
            {
            dim3 dg = dim3(48, 1, 1);
            dim3 db = dim3(288, 1, 1);
            kernelConvolutionCM<<<dg,db>>>(tabGMImageGris, tabGMConvolutionOutput, w, h, kernelSize/2);

            }

        //Convolution texture
        else if (version == Version::TEXTURE)
            {
            dim3 dg = dim3(14, 1, 1);
            dim3 db = dim3(1024, 1, 1);
            kernelConvolutionTexture<<<dg,db>>>(tabGMConvolutionOutput, w, h, kernelSize);
            }

    //MinMax
        {
        dim3 dg = dim3(48, 1, 1);
        dim3 db = dim3(512, 1, 1);
        size_t sizeSMMinMax = 2 * Device::nbThread(dg, db) * sizeof(uchar);
        kernelMinMax<<<dg, db, sizeSMMinMax>>>(tabGMConvolutionOutput, tabGMMinMax, w , h);
        //uchar *minMax = new uchar[2];
        //Device::memcpyDToH(minMax, tabGMMinMax, sizeof(uchar) * 2);
        }
        //Amplification
            {
            dim3 dg = dim3(48, 1, 1);
            dim3 db = dim3(576, 1, 1);
        kernelAmplification<<<dg, db>>>(tabGMConvolutionOutput, tabGMMinMax, w, h);
        }
            Device::memcpyDToD(ptrDevPixels, tabGMConvolutionOutput, sizeof(uchar) * w * h);
    }
else
    {
    ptrTabPixelVideoToGray();
    //OpenMP Convolution
        {
        openMPConvolution();
        }
	//MinMax
	{
	openMPMinMax();
        }
    //Amplification
	{
	amplificationOpenMP();
	}

                // Copy the final output to ptrDevPixel
                Device::memcpyHToD(ptrDevPixels, tabImageConvolutionOutput, sizeof(uchar) * w * h);

    }


    //ptrDevPixels= tabImageConvolutionOutput;
}

void Convolution::animationStep()
{
t++;

    //video
    {
    if (version != Version::FULL_LOAD)
	{
	free(ptrTabPixelVideo);
	}
    this->ptrTabPixelVideo = frameProvider.getFrame();
    }
}

void Convolution::ptrTabPixelVideoToGray()
{
//    float POIDS = 1. / 3.;
//    const int NB_THREAD = OmpTools::setAndGetNaturalGranularity();
//    #pragma omp parallel
//	{
//	const int TID = OmpTools::getTid();
//	const int WH = w * h;
//	int s = TID;
//	while (s < WH/4)
//	    {
//	    uchar4 color = ptrTabPixelVideo[s];
//	    uchar r = color.x;
//	    uchar g = color.y;
//	    uchar b = color.z;
//	    ptrTabPixelGray[s] =  125;
//	    s += NB_THREAD;
//	    }
//	}
float POIDS = 1. / 3.;
#pragma omp parallel for
for (int i = 0; i < w * h; i++)
{
uchar4 color = ptrTabPixelVideo[i];
uchar r = color.x;
uchar g = color.y;
uchar b = color.z;
ptrTabPixelGray[i] = r * POIDS + g * POIDS + b * POIDS;
}

}

void Convolution::openMPConvolution()
{

const int NB_THREAD = OmpTools::setAndGetNaturalGranularity();
#pragma omp parallel
{
int s = OmpTools::getTid();

while (s < w * h)
{
int i = 0;
int j = 0;
int u = 0;
int v = 0;
cpu::IndiceTools::toIJ(s, w, &u, &v);

float sum = 0;
int sizeLine = 2 * radius + 1;
while (i < sizeLine)
    {
    int x = (v - radius + i);
    while (j < sizeLine)
	{
	int y = u - radius + j;
	if (!(x < 0 || y < 0 || x >= w || y >= h))
	    {
	    sum += tabKernelConvolution[j * sizeLine + i] * ptrTabPixelGray[w * y + x];
	    }
	++j;
	}
    ++i;
    j = 0;
    }
if (sum < 0)
    {
    sum = 0;
    }
else if (sum > 255)
    {
    sum = 255;
    }
tabImageConvolutionOutput[s] = (int) sum;
s += NB_THREAD;
}
}
}

void Convolution::openMPMinMax()
    {
    int nbThread = OmpTools::setAndGetNaturalGranularity();
    uchar* tabMinMax = new uchar[nbThread * 2];
#pragma omp parallel
    {
	int TID = OmpTools::getTid();
	tabMinMax[TID] = tabImageConvolutionOutput[TID];
	tabMinMax[TID + nbThread] = tabImageConvolutionOutput[TID];
	int s = TID +nbThread;
	while(s<w*h)
	    {
		bool smaller = tabImageConvolutionOutput[s] < tabMinMax[TID];
		tabMinMax[TID] = tabImageConvolutionOutput[s]*(int)smaller + tabMinMax[TID]*((int)(!smaller));

		bool greater = tabImageConvolutionOutput[s] > tabMinMax[TID + nbThread];
		tabMinMax[TID + nbThread] = tabImageConvolutionOutput[s]*(int)greater + tabMinMax[TID +nbThread]*((int)(!greater));
	    s+= nbThread;
	    }
    }

for(int i = 1;i < nbThread; i++)
    {
    tabMinMax[0] = tabMinMax[i] < tabMinMax[0] ? tabMinMax[i] : tabMinMax[0];
    tabMinMax[nbThread] = tabMinMax[i +nbThread] > tabMinMax[nbThread] ? tabMinMax[i+nbThread] : tabMinMax[nbThread];
    }
tabMinMaxOmp[0] = tabMinMax[0];
tabMinMaxOmp[1] = tabMinMax[nbThread];
    }

void Convolution::amplificationOpenMP()
    {
    const int NB_THREAD = OmpTools::setAndGetNaturalGranularity();
    const int WH = w * h;
    const uchar min = tabMinMaxOmp[0];
    const uchar max = tabMinMaxOmp[1];
    cpu::Interval<uchar> input(min,max);
    cpu::Interval<uchar> output((uchar)0,(uchar)255);
    cpu::Calibreur<uchar> calibreur(input, output);
#pragma omp parallel
    {
	const int TID = OmpTools::getTid();

            //Calibreur<uchar> calibreur(min, max, (uchar)0, (uchar)255);
            int s = TID;
            while (s < WH)
        	{
        	if (min == max)
        	    tabImageConvolutionOutput[s] = (uchar)127;
        	else
        	    {
        	    calibreur.calibrer(&tabImageConvolutionOutput[s]);
        	    }
        	s += NB_THREAD;
        	}

    }

    }
