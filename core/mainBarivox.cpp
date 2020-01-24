#include <iostream>
#include <stdlib.h>

#include "ConvolutionProvider.h"
#include "AmplificationProvider.h"

#include "Barivox.h"

#include "Settings_GPU.h"
using namespace gpu;

using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |* Declaration *|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainBarivox(Settings& settings);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static void convolutionVideo();
static void imageVideo();

// Tools
template<typename T>
static void barivox(Provider_I<T>* ptrProvider, string titre, int nbIteration=1000);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainBarivox(Settings& settings)
    {
    cout << "\n[Barivox] mode" << endl;

    // Attention : Un a la fois seulement!

    convolutionVideo();
//    imageVideo();

    cout << "\n[Barivox] end" << endl;

    return EXIT_SUCCESS;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void convolutionVideo()
    {
    const int NB_ITERATION = 1000;

    ConvolutionProvider provider;
    barivox<uchar>(&provider, "ConvolutionVideo_GREY_uchar",NB_ITERATION);
    }

void imageVideo()
    {
    const int NB_ITERATION = 1000;

    AmplificationProvider provider;
    barivox<uchar4>(&provider, "ImageVideo_RGBA_uchar4",NB_ITERATION);
    }

/*-----------------------------------*\
 |*		Tools	        	*|
 \*-----------------------------------*/

/**
 * Grid 1d Only
 */
template<typename T>
void barivox(Provider_I<T>* ptrProvider, string titre,int nbIteration)
    {
    cout << "\n[Barivox] : " << titre << endl;

    // Define Grid
    int mp = Device::getMPCount();
    int coreMp = Device::getCoreCountMP();
    int nbThreadBlockMax = Device::getMaxThreadPerBlock();
    int warpSize = Device::getWarpSize();

    dim3 dgStart(mp * 2, 1, 1);
    VariateurData variateurDg(mp, 8 * mp, mp); 				// (min,max,step)  Attention : A definir intelligement selon le GPU !
    VariateurData variateurDb(coreMp, nbThreadBlockMax, warpSize); 	// (min,max,step)  Attention : A definir intelligement selon le GPU !

    // Run
    const bool IS_ANIMATOR_VERBOSITY_ENABLE = false; //TODO CBI BarivoxOption
    const bool IS_BARIVOX_VERBOSITY_ENABLE = true;


    Barivox<T> barivox(ptrProvider, dgStart, variateurDg, variateurDb, nbIteration, IS_BARIVOX_VERBOSITY_ENABLE, IS_ANIMATOR_VERBOSITY_ENABLE);
    const BarivoxOutput output = barivox.run();

    // Print
	{
	cout << output << endl;
	}

    // Save
	{
	output.save("out/barivox/barivox_" + titre);
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
