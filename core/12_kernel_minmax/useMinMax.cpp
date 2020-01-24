#include <iostream>
#include "Grid.h"
#include "Device.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cudaTools.h"

#include "host/MinMax.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__host__ bool isMinMax0To255Ok();
__host__ bool isMinMax23To177Ok();
__host__ bool isMinMaxLoopOk();
__host__ bool isMinMaxUniColorOk();
__host__ bool isMinMaxTestOk(const Grid& grid, uchar* tabGM, int const& w, int const& h);

__host__ uchar* buildRandomArray(uchar, uchar, int);
__host__ uchar* buildSameArray(uchar, int);
__host__ void replaceMinMaxAtRandom(uchar*, int);
__host__ void replaceMinMaxAtBeginningEnd(uchar*, int);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__host__ bool isMinMaxTestOk(const Grid& grid, uchar* tabGM, int const& w, int const& h, uchar min, uchar max)
    {

    uchar* result = (uchar*) malloc(2 * sizeof(uchar));
    MinMax minMax(grid, tabGM, result, w, h);
    minMax.run();

    cout << "min attendu : " << (int)min << " min trouvé : " << (int)result[0] << endl;
    cout << "max attendu : " << (int)max << " max trouvé : " << (int)result[1] << endl;

    bool isOk = (result[0] == min && result[1] == max);

    return isOk;
    }

__host__ bool isMinMaxConvolutionOk(const Grid& grid, uchar* tabGM, int const& w, int const& h, uchar min, uchar max)
    {

    uchar* result = (uchar*) malloc(2 * sizeof(uchar));
    MinMax minMax(grid, tabGM, result, w, h);
    minMax.run();

    cout << "min : " << (int)result[0] << " - Expected : " << (int)min << endl;
    cout << "max : " << (int)result[1] << " - Expected : " << (int)max << endl;

    bool isOk = (result[0] == min && result[1] == max);

    return isOk;
    }

__host__ bool isMinMax0To255Ok()
    {
    dim3 dg = dim3(64, 1, 1);
    dim3 db = dim3(128, 1, 1);
    Grid grid(dg, db);

    int w = 1000;
    int h = 1000;

    uchar* tabGM = buildRandomArray(1, 255, w*h);
    replaceMinMaxAtRandom(tabGM, w*h);

    uchar min = 0;
    uchar max = 255;

    cout << "isMinMax0To255Ok" << endl;
    return isMinMaxTestOk(grid, tabGM, w, h, min, max);
    }

__host__ bool isMinMax23To177Ok()
    {
    dim3 dg = dim3(64, 1, 1);
    dim3 db = dim3(128, 1, 1);
    Grid grid(dg, db);

    int w = 1000;
    int h = 1000;

    cout << "isMinMax23To177Ok" << endl;
    uchar* tabGM = buildRandomArray(23, 178, w*h);

    uchar min = 23;
    uchar max = 177;

    return isMinMaxTestOk(grid, tabGM, w, h, min, max);
    }

__host__ bool isMinMaxLoopOk()
    {
    dim3 dg = dim3(64, 1, 1);
    dim3 db = dim3(128, 1, 1);
    Grid grid(dg, db);

    int w = 1000;
    int h = 1000;

    cout << "isMinMaxLoopOk" << endl;
    uchar* tabGM = buildRandomArray(1, 255, w*h);
    replaceMinMaxAtBeginningEnd(tabGM, w*h);

    uchar min = 0;
    uchar max = 255;

    return isMinMaxTestOk(grid, tabGM, w, h, min, max);
    }

__host__ bool isMinMaxUniColorOk()
    {
    srand(time(0));

    dim3 dg = dim3(64, 1, 1);
    dim3 db = dim3(128, 1, 1);
    Grid grid(dg, db);

    int w = 1000;
    int h = 1000;

    uchar value = rand() % 255;

    cout << "isMinMaxUniColorOk" << endl;
    uchar* tabGM = buildSameArray(value, w*h);

    return isMinMaxTestOk(grid, tabGM, w, h, value, value);
    }

__host__ uchar* buildRandomArray(uchar min, uchar max, int sizeArray)
    {
    srand(time(0));

    uchar* tab = (uchar*) malloc(sizeArray * sizeof(uchar));

    for (int i = 0; i < sizeArray; i++)
	{
	tab[i] = (rand() % (max - min)) + min;
	}

    return tab;
    }

__host__ uchar* buildSameArray(uchar value, int sizeArray)
    {
    uchar* tab = (uchar*) malloc(sizeArray * sizeof(uchar));

    for (int i = 0; i < sizeArray; i++)
	{
	tab[i] = value;
	}

    return tab;
    }

__host__ void replaceMinMaxAtRandom(uchar* tabValue, int sizeArray)
    {
    srand(time(0));
    int x = rand() % sizeArray;
    int y = rand() % sizeArray;
    cout << x << " - " << y  << " - " << sizeArray<< endl;
    tabValue[x] = 0;
    tabValue[y] = 255;
    }

__host__ void replaceMinMaxAtBeginningEnd(uchar* tabValue, int sizeArray)
    {
    tabValue[0] = 0;
    tabValue[sizeArray - 1] = 255;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

