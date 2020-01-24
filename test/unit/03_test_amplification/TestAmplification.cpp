#include "TestAmplification.h"

#include <cudaTools.h>
#include <Device.h>
#include <Grid.h>

#include "../13_kernel_amplification/provider/AmplificationProvider.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/
extern uchar* testTabGMValueAfterAmplification(Grid grid, uint min, uint max, int w, int h);
/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Constructor		*|
 \*-------------------------------------*/

TestAmplification::TestAmplification()
    {
    TEST_ADD(TestAmplification::testAmplificationGrid);
    }

/*--------------------------------------*\
 |*		Methodes		*|
 \*-------------------------------------*/

void TestAmplification::testAmplificationGrid(void)
    {
    uint min = 100;
    uint max = 101;
    int w = 10;
    int h = 10;

    const int NB_DEVICE = Device::getDeviceCount();

    //#pragma omp paralle for
    for (int deviceId = 0; deviceId < NB_DEVICE; deviceId++)
	{
	Device::setDevice(deviceId);

	// Dépend du device
	int MP = Device::getDeviceCount(); // 24 M6000
	int MAX_THEAD_BLOCK = Device::getMaxThreadPerBlock(); // 1024
	int MP_MAX = 96; // disons

	for (int g = MP; g <= MP_MAX; g = g * 2)
	    {
	    for (int b = 64; b <= MAX_THEAD_BLOCK; b = b * 2)
		{
		dim3 dg = dim3(g, 1, 1); // 1D
		dim3 db = dim3(b, 1, 1); // 1D
		Grid grid(dg, db);

		uchar* tabImage = AmplificationProvider::testTabGMValueAfterAmplification(grid, min, max, w, h);
		TEST_ASSERT(tabImage[0] == 0);
		TEST_ASSERT(tabImage[w * h - 1] == 255);
		}
	    }
	}

    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

