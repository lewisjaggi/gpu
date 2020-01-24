#pragma once

#include <Animable_I_GPU.h>
#include <cudaTools.h>
#include <Grid.h>

using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class AmplificationTest: public Animable_I<uchar4>
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	AmplificationTest(const Grid& grid, uint w, uint h, int min, int max);
	virtual ~AmplificationTest(void);

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*-------------------------*\
	|*   Override Animable_I   *|
	 \*------------------------*/

	/**
	 * Call periodicly by the api
	 */
	virtual void process(uchar4* ptrDevPixels, uint w, uint h, const DomaineMath& domaineMath);

	/**
	 * Call periodicly by the api
	 */
	virtual void animationStep();

	uchar* testTabGMValueAfterAmplificationUchar();
	float* testTabGMValueAfterAmplificationFloat();

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:
	int min;
	int max;
	uchar* tabGMIntervalleDevUchar;
	uchar* tabGMDevUchar;
	float* tabGMIntervalleDevFloat;
	float* tabGMDevFloat;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
