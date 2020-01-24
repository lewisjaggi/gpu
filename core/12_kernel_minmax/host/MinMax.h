#pragma once

#include "cudaTools.h"
#include "Grid.h"
/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class MinMax
    {
    public:
	MinMax(const Grid& grid, uchar* tabGM,  uchar* ptrResult, int const& w, int const& h);
	virtual ~MinMax();
	void run();

    private:
	dim3 dg;
	dim3 db;

	int const& w;
	int const& h;

	size_t sizeTabSM;
	size_t sizePtrDevResult;
	size_t sizeTabDevGM;

	uchar* tabGM;
	uchar* tabDevGM;
	uchar* ptrResult;
	uchar* ptrDevResult;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
