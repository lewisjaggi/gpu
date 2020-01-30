#include <stdio.h>
#include <Indice2D.h>
#include <cudaTools.h>
#include <ReductionMinMaxTools.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void kernelMinMax(uchar* tabGM, uchar* ptrDevResult, int w, int h);
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__device__ void reductionIntraThread(uchar* tabGM, uchar* tabSMMin, uchar* tabSMMax, int w, int h);
__device__ void reductionIntraThreadIf(uchar* tabGM, uchar* tabSMMin, uchar* tabSMMax, int w, int h);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * SizeSM must be twice the size of the Shared Memory !
 */

__global__ void kernelMinMax(uchar* tabGM, uchar* ptrDevResult, int w, int h)
    {

    extern __shared__ uchar tabSM[];
    uchar* tabSMMin = tabSM;
    uchar* tabSMMax = tabSM + (Indice2D::nbThreadLocal() * sizeof(uchar));

    // INTRA
    reductionIntraThread(tabGM, tabSMMin, tabSMMax, w, h);

    ptrDevResult[0] = 255;
    ptrDevResult[1] = 0;

    // SYNC
    __syncthreads();

    // TOOLS
    ReductionMinMaxTools::reductionMinMax(ptrDevResult, tabSMMin, tabSMMax);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__device__ void reductionIntraThread(uchar* tabGM, uchar* tabSMMin, uchar* tabSMMax, int w, int h)
    {
    const int NB_THREADS = Indice2D::nbThreadLocal();
    const int TID = Indice2D::tidLocal();
    const int limit = w * h;

    int s = TID;

    tabSMMin[TID] = tabGM[s];
    tabSMMax[TID] = tabGM[s];
    s += NB_THREADS;

    while (s < limit)
	{
	bool smaller = tabGM[s] < tabSMMin[TID];
	tabSMMin[TID] = tabGM[s]*(int)smaller + tabSMMin[TID]*((int)(!smaller));

	bool greater = tabGM[s] > tabSMMax[TID];
	tabSMMax[TID] = tabGM[s]*(int)greater + tabSMMax[TID]*((int)(!greater));

	s += NB_THREADS;
	}
    }


__device__ void reductionIntraThreadIf(uchar* tabGM, uchar* tabSMMin, uchar* tabSMMax, int w, int h)
    {
    const int NB_THREADS = Indice2D::nbThreadLocal();
    const int TID = Indice2D::tidLocal();
    const int limit = w * h;

    int s = TID;

    tabSMMin[TID] = tabGM[s];
    tabSMMax[TID] = tabGM[s];
    s += NB_THREADS;

    while (s < limit)
	{
	if(tabGM[s] < tabSMMin[TID])
	    {
	    tabSMMin[TID] = tabGM[s];
	    }

	if(tabGM[s] > tabSMMax[TID])
	    {
	    tabSMMax[TID] = tabGM[s];
	    }
	s += NB_THREADS;
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

