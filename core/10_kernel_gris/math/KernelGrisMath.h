#pragma once

#include <math.h>
#include "MathTools.h"

#include "ColorTools_GPU.h"
using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class KernelGrisMath
    {

	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__ KernelGrisMath(int w, int h)
	    {
	    this->POIDS = 1 / (double) 3;
	    }

	// constructeur copie automatique car pas pointeur dans VagueMath

	__device__
	    virtual ~KernelGrisMath()
	    {
	    // rien
	    }

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	__device__
	uchar colorIJ(uchar4* ptrColor, int i, int j)
	    {
	    uchar r = ptrColor->x;
	    uchar g = ptrColor->y;
	    uchar b = ptrColor->z;

	    uchar levelGris = r * POIDS + g * POIDS + b * POIDS;

	    return levelGris;
	    }

    private:

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Tools
	float POIDS;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
