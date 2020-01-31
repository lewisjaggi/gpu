#include <assert.h>
#include <Device.h>
#include <LaunchMode.h>
#include <Options.h>
#include <stdlib.h>
#include <Settings_GPU.h>
#include <iostream>

#include "cudaTools.h"

using namespace gpu;
using std::string;
using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern int mainImage(Settings& settings);
extern int mainAnimable(Settings& settings);
extern int mainBrutForce(Settings& settings);
extern int mainBarivox(Settings& settings);
extern int mainTest();

extern int mainTest();

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int main(int argc, char** argv);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static int use(Settings& settings, Options& option);
static int start(Settings& settings, Options& option);
static void initCuda(Settings& settings, Options& option);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int main(int argc, char** argv)
    {
    // Server Cuda2: in [0,3]	(4 Devices)
    // Server Cuda3: in [0,2]	(2 Devices)
    int DEVICE_ID = 0;
    bool IS_TEST = false;

    //WARNING Pour lancer les tests du noyau de convolution, il faut modifier
    // la constante KERNEL_CONVOLUTION_SIZE dans le fichier kernelConvolutionDevice

    LaunchMode launchMode = LaunchMode::ANIMABLE; // IMAGE  ANIMABLE  BARIVOX FORCEBRUT

    //DEVICE_ID is store 2 times but we haven't found any other option
    Settings settings(launchMode, DEVICE_ID, argc, argv);
    Options option(IS_TEST, DEVICE_ID);

    return use(settings, option);
    }

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int use(Settings& settings, Options& option)
    {
    initCuda(settings, option);
    int isOk = start(settings, option);

    Device::reset();
    return isOk;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void initCuda(Settings& settings, Options& option)
    {
    int deviceId = settings.getDeviceId();

    // Choose current device  (state of host-thread)
    Device::setDevice(deviceId);
    assert(Device::isCuda());

    // It can be usefull to preload driver, by example to practice benchmarking! (sometimes slow under linux)
    Device::loadCudaDriver(deviceId);
    // Device::loadCudaDriverAll();// Force driver to be load for all GPU
    }

int start(Settings& settings, Options& option)
    {
    // print
	{
	// Device::printAll();
	Device::printAllSimple();
	Device::printCurrent();
	//Device::print(option.getDeviceId());
	}

    if (option.isTest())
	{
	return mainTest();
	}
    else
	{
	switch (settings.getLauchMode())
	    {
	    case IMAGE:
		return mainImage(settings);
	    case ANIMABLE:
		return mainAnimable(settings);
	    case FORCEBRUT:
		return mainBrutForce(settings);
	    case BARIVOX:
		return mainBarivox(settings);
	    default:
		return mainImage(settings);
	    }
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

