#include "AmplificationTest.h"

#include <Device.h>
#include <DeviceWrapperTemplate.h>
#include <iostream>

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void generateTestImage(uchar4* image,uchar* tabGM, int min, int max, uint w, uint h);

extern __global__ void copyTabGMToImage(uchar4* image,uchar* tabGM,uint w, uint h);

extern __global__ void kernelAmplification(uchar* tabGM, uchar* tabGMIntervalle, int w, int h);

extern __global__ void generateTestImage(uchar4* image,float* tabGM, int min, int max, uint w, uint h);

extern __global__ void copyTabGMToImage(uchar4* image,float* tabGM, uint w, uint h);

extern __global__ void kernelAmplification(float* tabGM, float* tabGMIntervalle, int w, int h);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

AmplificationTest::AmplificationTest(const Grid& grid, uint w, uint h, int min, int max) :
	Animable_I<uchar4>(grid, w, h, "AmplificationTest_Cuda_RGBA_uchar4")
    {
    this->min = min;
    this->max = max;

    uchar tabGMIntervalleUchar[] =
	{
	(uchar) min, (uchar) max
	};

    float tabGMIntervalleFloat[] =
	{
	(float) min, (float) max
	};

    Device::malloc(&tabGMIntervalleDevUchar, (size_t)(sizeof(uchar) * 2));

    Device::malloc(&tabGMIntervalleDevFloat, (size_t)(sizeof(float) * 2));

    Device::memcpyHToD(tabGMIntervalleDevUchar, tabGMIntervalleUchar, (size_t)(sizeof(uchar) * 2));

    Device::memcpyHToD(tabGMIntervalleDevFloat, tabGMIntervalleFloat, (size_t)(sizeof(float) * 2));

    Device::malloc(&tabGMDevUchar, (size_t)(sizeof(uchar) * h * w));

    Device::malloc(&tabGMDevFloat, (size_t)(sizeof(float) * h * w));
    }

AmplificationTest::~AmplificationTest()
    {
    Device::free(tabGMIntervalleDevUchar);
    Device::free(tabGMDevUchar);
    Device::free(tabGMIntervalleDevFloat);
    Device::free(tabGMDevFloat);
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

/**
 * Override
 * Call periodicly by the API
 */
void AmplificationTest::process(uchar4* ptrDevPixels, uint w, uint h, const DomaineMath& domaineMath)
    {
    //generateTestImage<<<dg,db>>>(ptrDevPixels,tabGMDevUchar,min,max, w, h);
    //kernelAmplification<<<dg,db>>>(tabGMDevUchar,tabGMIntervalleDevUchar,w, h);
    //copyTabGMToImage<<<dg,db>>>(ptrDevPixels,tabGMDevUchar,w, h);
generateTestImage<<<dg,db>>>(ptrDevPixels,tabGMDevFloat,min,max, w, h);
kernelAmplification<<<dg,db>>>(tabGMDevFloat,tabGMIntervalleDevFloat,w, h);
copyTabGMToImage<<<dg,db>>>(ptrDevPixels,tabGMDevFloat,w, h);
}

/**
 * Override
 * Call periodicly by the API
 */
void AmplificationTest::animationStep()
{
//rien
}

uchar* AmplificationTest::testTabGMValueAfterAmplificationUchar()
{
uchar4* fakeImage = new uchar4[w * h];
uchar4* fakeImageDev;

Device::malloc(&fakeImageDev, (size_t)(sizeof(uchar4) * w * h));

Device::memcpyHToD(fakeImageDev, fakeImage, (size_t)(sizeof(uchar4) * w * h));

generateTestImage<<<dg,db>>>(fakeImageDev,tabGMDevUchar,min,max, w, h);
kernelAmplification<<<dg,db>>>(tabGMDevUchar,tabGMIntervalleDevUchar,w, h);

uchar* tabGMtest = new uchar[w * h];

Device::memcpyDToH(tabGMtest, tabGMDevUchar, (size_t)(sizeof(uchar) * w * h));

Device::free(fakeImageDev);

return tabGMtest;
}

float* AmplificationTest::testTabGMValueAfterAmplificationFloat()
{
    uchar4* fakeImage = new uchar4[w * h];
    uchar4* fakeImageDev;

    Device::malloc(&fakeImageDev, (size_t)(sizeof(uchar4) * w * h));

    Device::memcpyHToD(fakeImageDev, fakeImage, (size_t)(sizeof(uchar4) * w * h));

    generateTestImage<<<dg,db>>>(fakeImageDev,tabGMDevFloat,min,max, w, h);
    kernelAmplification<<<dg,db>>>(tabGMDevFloat,tabGMIntervalleDevFloat,w, h);

    float* tabGMtest = new float[w * h];

    Device::memcpyDToH(tabGMtest, tabGMDevFloat, (size_t)(sizeof(float) * w * h));

    Device::free(fakeImageDev);

    return tabGMtest;
}
