#include <cudaTools.h>
#include <Indice2D.h>
#include <IndiceTools_GPU.h>

using namespace gpu;

 /*---------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

template <typename T>
__device__ void generateTestImage(uchar4* image,T* tabGM, int min, int max, uint w, uint h);

__global__ void generateTestImage(uchar4* image,uchar* tabGM, int min, int max, uint w, uint h);

__global__ void generateTestImage(uchar4* image,float* tabGM, int min, int max, uint w, uint h);

template <typename T>
__device__ void copyTabGMToImage(uchar4* image,T* tabGM,uint w, uint h);

__global__ void copyTabGMToImage(uchar4* image,uchar* tabGM,uint w, uint h);

__global__ void copyTabGMToImage(uchar4* image,float* tabGM,uint w, uint h);

 /*---------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

template <typename T>
__device__ void generateTestImage(uchar4* image,T* tabGM, int min, int max, uint w, uint h)
    {
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int WH = w * h;

    int valGris = min;

    float coeff=(float)(max-min+1)/(float)h; //permet de savoir quelle valeur de gris il faut à quelle ligne de l'image (relation linéaire ou la 1ere ligne a la valeur min et la derniere la valeur max)

    int s = TID;

    int i;
    int j;

    while (s < WH)
	{
	IndiceTools::toIJ(s, w, &i, &j);

	valGris=coeff*(float)i+min;

	image[s].x = valGris;
	image[s].y = valGris;
	image[s].z = valGris;
	image[s].w = 255; // opacity facultatif

	tabGM[s]=(T)valGris;

	s += NB_THREAD;
	}
    }

__global__ void generateTestImage(uchar4* image,uchar* tabGM, int min, int max, uint w, uint h){
    generateTestImage<uchar>(image,tabGM,min,max,w,h);
}

__global__ void generateTestImage(uchar4* image,float* tabGM, int min, int max, uint w, uint h) {
    generateTestImage<float>(image,tabGM,min,max,w,h);
}

template <typename T>
__device__ void copyTabGMToImage(uchar4* image,T* tabGM,uint w, uint h)
    {
    const int TID = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int WH = w * h;

    int s=TID;

    int valGris;
    while (s < WH)
	{
	valGris=(uchar)tabGM[s];

	image[s].x = valGris;
	image[s].y = valGris;
	image[s].z = valGris;
	image[s].w = 255; // opacity facultatif

	s += NB_THREAD;
	}
    }


__global__ void copyTabGMToImage(uchar4* image,uchar* tabGM,uint w, uint h){
    copyTabGMToImage<uchar>(image,tabGM,w,h);
}

__global__ void copyTabGMToImage(uchar4* image,float* tabGM,uint w, uint h){
    copyTabGMToImage<float>(image,tabGM,w,h);
}


