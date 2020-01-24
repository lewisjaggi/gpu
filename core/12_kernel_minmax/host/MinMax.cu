#include "MinMax.h"

#include <assert.h>
#include <curand_kernel.h>
#include <Device.h>
#include <DeviceWrapperTemplate.h>
#include <iostream>


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/


extern __global__ void kernelMinMax(uchar* tabGM, uchar* ptrDevResult, int w, int h);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Constructeur			*|
 \*-------------------------------------*/
MinMax::MinMax(const Grid& grid, uchar* tabGM,  uchar* ptrResult, int const& w, int const& h) :
tabGM(tabGM), ptrResult(ptrResult), w(w), h(h)
    {
    // Grid
    this->dg = grid.dg;
    this->db = grid.db;

    // double size for tabMinSM and tabMaxSM
    this->sizeTabSM = 2 * sizeof(uchar) * grid.threadByBlock();

    this->sizePtrDevResult = 2 * sizeof(uchar);
    this->sizeTabDevGM = w * h * sizeof(uchar);

    this->ptrDevResult = NULL;
    Device::malloc(&ptrDevResult, sizePtrDevResult);
    Device::memclear(ptrDevResult, sizePtrDevResult);

    this->tabDevGM = NULL;
    Device::malloc(&tabDevGM, sizeTabDevGM);
    Device::memclear(tabDevGM, sizeTabDevGM);
    Device::memcpyHToD(tabDevGM, tabGM, sizeTabDevGM);
    }

MinMax::~MinMax(void)
    {
    Device::free(ptrDevResult);
    }

/*--------------------------------------*\
 |*		Methode			*|
 \*-------------------------------------*/

void MinMax::run()
    {
	kernelMinMax<<<dg, db, sizeTabSM>>>(tabDevGM, ptrDevResult, w, h);
	Device::memcpyDToH(ptrResult, ptrDevResult, sizePtrDevResult);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
