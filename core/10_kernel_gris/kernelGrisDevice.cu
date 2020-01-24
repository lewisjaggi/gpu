#include "Indice2D.h"
#include "cudaTools.h"

#include <cuda.h>

#include "Device.h"

#include "KernelGrisMath.h"

#include "IndiceTools_GPU.h"
using namespace gpu;



__global__ void kernelGris(uchar4* tabGMImageCouleur, uchar* tabGMImageGris, uint w, uint h);


__global__ void kernelGris(uchar4* tabGMImageCouleur, uchar* tabGMImageGris, uint w, uint h)
    {
    KernelGrisMath math = KernelGrisMath(w, h);

    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int WH=w*h;

    int pixelI;	// in [0,h[
    int pixelJ; // in [0,w[

    int s = TID;
    while (s < WH)
	{
	IndiceTools::toIJ(s, w, &pixelI, &pixelJ); 	// update (pixelI, pixelJ)

	tabGMImageGris[s] = math.colorIJ(&tabGMImageCouleur[s],pixelI, pixelJ);
	s += NB_THREAD;
	}
    }


