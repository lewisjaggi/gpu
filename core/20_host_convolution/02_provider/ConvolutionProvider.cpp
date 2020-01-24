#include "Convolution.h"
#include "ConvolutionProvider.h"

#include "MathTools.h"
#include "Grid.h"
#include "NoyauCreator.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static string nameVideo();

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Override
 */
Animable_I<uchar>* ConvolutionProvider::createAnimable()
    {
    // video
    string videoName = nameVideo();

    // Grid Cuda
    int mp = Device::getMPCount();
    int coreMP = Device::getCoreCountMP();

    dim3 dg = dim3(mp, 2, 1);  	  // disons a optimiser, depend du gpu
    dim3 db = dim3(coreMP, 2, 1); // disons a optimiser, depend du gpu
    Grid grid(dg, db);

    // Dimension : autoroute
    uint w = 1920;
    uint h = 1080;

    NoyauCreator* noyauCreator = new NoyauCreator();

    return new Convolution(grid, w, h, videoName, noyauCreator->getTabNoyau(), 9);//TODO: Add kernel convolution
    }

/**
 * Override
 */
Image_I* ConvolutionProvider::createImageGL(void)
    {
    ColorRGB_01 colorTexte(1, 0, 0); // red
    return new ImageAnimable_GRAY_uchar(createAnimable(), colorTexte);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

string nameVideo()
    {
    // Important de tester en haute definition (autoroute.mp4) pour vallider la dll ffmpeg
    // petite video ne l'utilise pas!

#ifdef _WIN32
    // Work
	{
	//return "C:\\Users\\cedric.bilat\\Desktop\\neilPryde.avi";// ok
	//return"C:\\Users\\cedric.bilat\\Desktop\\autoroute.mp4";//ok
	}

    // Home
	{
	//  return "C:\\Users\\bilat\\Desktop\\neilPryde.avi"; // ok
	return "C:\\Users\\bilat\\Desktop\\autoroute.mp4";// ok
	}
//#elif  __arm__
#else
	{
	 // return "/opt/cbi/data/video/neilPryde.avi"; // ok
	return "/opt/cbi/data/video/autoroute.mp4"; // ok
	}
#endif
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
