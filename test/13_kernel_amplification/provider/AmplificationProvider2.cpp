#include "AmplificationProvider2.h"

#include <Animable_I_GPU.h>
#include <ColorRGB_01.h>
#include <Device.h>
#include <Grid.h>
#include <Image_I.h>
#include <ImageAnimable_GPU.h>

#include "../host/AmplificationTest2.h"


Animable_I<uchar4>* AmplificationProvider2::createAnimable()
    {

    // Dimension
    int w = 16 * 80;
    int h = 16 * 60;

    // grid cuda
    int mp = Device::getMPCount();
    int coreMP = Device::getCoreCountMP();

    std::cout << mp << " " << coreMP << std::endl;
    dim3 dg = dim3(mp, 16, 1);  		// disons, a optimiser selon le gpu, peut drastiqument ameliorer ou baisser les performances
    dim3 db = dim3(coreMP, 4, 1);   	// disons, a optimiser selon le gpu, peut drastiqument ameliorer ou baisser les performances

    Grid grid(dg, db);


    return new AmplificationTest2(grid, w, h,91,101);

    }


Image_I* AmplificationProvider2::createImageGL(void)
    {
    ColorRGB_01 colorTexte(0, 1, 0); // Green
    return new ImageAnimable_RGBA_uchar4(createAnimable(), colorTexte);
    }


uchar* AmplificationProvider2::testTabGMValueAfterAmplification(Grid grid, uint min, uint max, int w, int h)
    {
    AmplificationTest2 ampliTest(grid, w, h,min,max);
    return ampliTest.testTabGMValueAfterAmplificationUchar();
    }
