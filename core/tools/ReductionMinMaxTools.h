#pragma once

#include "cudaTools.h"
#include "Lock.h"

__device__ int mutex = 0;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

class ReductionMinMaxTools
    {
    public:

	static __device__ void reductionMinMax(uchar* ptrTabResultGM, uchar* ptrTabSMMin, uchar* ptrTabSMMax)
	    {
	    Lock lock = Lock(&mutex);

	    reductionIntraBlock(ptrTabSMMin, ptrTabSMMax);
	    __syncthreads();
	    reductionInterblock(ptrTabResultGM, ptrTabSMMin, ptrTabSMMax, &lock);
	    }

    private:

	/*--------------------------------------*\
	|*	reductionIntraBlock		*|
	 \*-------------------------------------*/

	static __device__ void reductionIntraBlock(uchar* tabSMMin, uchar* tabSMMax)
	    {
	    // Ecrasement sucessifs dans une boucle (utiliser methode ecrasement ci-dessus)
	    int middle = blockDim.x;
	    const int TID_LOCAL = threadIdx.x;

	    do {
		middle = middle >> 1;
		if(TID_LOCAL < middle)
		    {
		    tabSMMin[TID_LOCAL] = tabSMMin[TID_LOCAL] < tabSMMin[TID_LOCAL+middle] ? tabSMMin[TID_LOCAL] : tabSMMin[TID_LOCAL+middle];
		    tabSMMax[TID_LOCAL] = tabSMMax[TID_LOCAL] > tabSMMax[TID_LOCAL+middle] ? tabSMMax[TID_LOCAL] : tabSMMax[TID_LOCAL+middle];
		    __syncthreads(); // pour touts les threads d'un meme block
		    }
		}
	    while(middle > 0);

	    }

	/*--------------------------------------*\
	|*	reductionInterblock		*|
	 \*-------------------------------------*/

	static __device__ void reductionInterblock(uchar* ptrDevResultatGM, uchar* tabSMMin, uchar* tabSMMax, Lock* ptrLock)
	    {
	    if(threadIdx.x == 0)
		{
		ptrLock->lock();
		ptrDevResultatGM[0] = ptrDevResultatGM[0] < tabSMMin[0] ? ptrDevResultatGM[0] : tabSMMin[0];
		ptrDevResultatGM[1] = ptrDevResultatGM[1] > tabSMMax[0] ? ptrDevResultatGM[1] : tabSMMax[0];
		ptrLock->unlock();
		}
	    }

    };

/*----------------------------------------------------------------------*\
|*			End	 					*|
 \*---------------------------------------------------------------------*/
