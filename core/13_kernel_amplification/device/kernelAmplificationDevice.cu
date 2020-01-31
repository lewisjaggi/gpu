#include <cudaTools.h>
#include <Calibreur_GPU.h>
#include <Indice2D.h>
#include <Interval_GPU.h>
#include <stdlib.h>

using namespace gpu;
using namespace std;

/*---------------------------------------------------------------------*\
|*			Declaration 					*|
 \*---------------------------------------------------------------------*/

template <typename T>
//__device__ void kernelAmplification(T* tabGM, T* tabGMIntervalle, int w, int h);

__global__ void kernelAmplification(uchar *tabGM, uchar *tabGMIntervalle, int w, int h);
__global__ void kernelAmplification(float *tabGM, float *tabGMIntervalle, int w, int h);

/*---------------------------------------------------------------------*\
|*			Implementation 					*|
 \*---------------------------------------------------------------------*/

//template <typename T>
//__device__ void kernelAmplification(T* tabGM, T* tabGMIntervalle, int w, int h)
//    {
//    const int TID = Indice2D::tid();
//    const int NB_THREAD = Indice2D::nbThread();
//    const int WH = w * h;
//    const T min = tabGMIntervalle[0];
//    const T max = tabGMIntervalle[1];
//
//    Calibreur<T> calibreur(min, max, (T) 0, (T) 255);
//    int s = TID;
//    while (s < WH)
//	{
//	if(min==max)
//	tabGM[s]=(T)127;
//	else
//	    {
//	    calibreur.calibrer(&tabGM[s]);
//	    }
//	s += NB_THREAD;
//	}
//
//    }

__global__ void kernelAmplification(uchar *tabGM, uchar *tabGMIntervalle, int w, int h)
    {
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int WH = w * h;
    const uchar min = tabGMIntervalle[0];
    const uchar max = tabGMIntervalle[1];

   Interval<uchar> input(min,max);
   Interval<uchar> output((uchar)0,(uchar)255);

    Calibreur<uchar> calibreur(input, output);
    //Calibreur<uchar> calibreur(min, max, (uchar)0, (uchar)255);
    int s = TID;
    while (s < WH)
	{
//	if (min == max)
//	    tabGM[s] = (uchar)127;
//	else
//	    {
	    calibreur.calibrer(&tabGM[s]);
//	    }
	s += NB_THREAD;
	}

    }

__global__ void kernelAmplification(float *tabGM, float *tabGMIntervalle, int w, int h)
    {
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int WH = w * h;
    const float min = tabGMIntervalle[0];
    const float max = tabGMIntervalle[1];

   Interval<float> input(min,max);
   Interval<float> output((float)0,(float)255);

    Calibreur<float> calibreur(input, output);
    //Calibreur<uchar> calibreur(min, max, (uchar)0, (uchar)255);
    int s = TID;
    while (s < WH)
	{
	if (min == max)
	    tabGM[s] = (float)127;
	else
	    {
	    calibreur.calibrer(&tabGM[s]);
	    }
	s += NB_THREAD;
	}
    }
