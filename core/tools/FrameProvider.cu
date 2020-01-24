#include <assert.h>
#include <iostream>


#include "OmpTools.h"
#include "OpencvTools_GPU.h"
#include "FrameProvider.h"
#include <thread>
#include <chrono>
#include "Device.h"

using std::cout;
using std::cerr;
using std::endl;


FrameProvider::FrameProvider(uint w, uint h, string nameVideo, Version version) :
	nameVideo(nameVideo), capture(nameVideo), matRGBA(h, w, CV_8UC1), version(version), w(w), h(h)

    {

    //video
	{
	bool isOk = capture.start();
	if (!isOk)
	    {
	    cerr << "[ConvolutionVideo] : failed to open : " << nameVideo << endl;
	    exit (EXIT_FAILURE);
	    }
	assert(capture.getW() == w && capture.getH() == h);
	}


	if (version == Version::FULL_LOAD)
	    {
	    for (int i = 0; i < FRAME_NUMBER; i++)
		{
		fullVideo[i] = loadFrame();
		}

	    }

    }

FrameProvider::~FrameProvider()
    {
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

uchar4* FrameProvider::loadFrame()
    {
    //video
    	{
    	Mat matBGR = capture.provideBGR();

    	OpencvTools_GPU::switchRB(this->matRGBA, matBGR);

    	uchar4* newImage = OpencvTools_GPU::castToUchar4(matRGBA);
    	uchar4* img = (uchar4*)malloc(w * h * sizeof(uchar4));
    	memcpy(img, newImage, w * h * sizeof(uchar4));

    	return img;
    	}
    }

uchar4* FrameProvider::getFrame()
    {
	if (version == Version::FULL_LOAD)
	    {
	    uchar4* frame = fullVideo[currentFrame];
	    currentFrame++;
	    if (currentFrame >= FRAME_NUMBER)
		{
		currentFrame = 0;
		}
	    return frame;
	    }
	else
	    {
	    return loadFrame();
	    }
    }


