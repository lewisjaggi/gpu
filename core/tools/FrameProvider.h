#pragma once

#include <iostream>

#include <boost/lockfree/spsc_queue.hpp>
#include "cudaTools.h"
#include "CVCaptureVideo.h"
#include "Version.h"


using cv::Mat;
using std::string;

#ifndef Fifo
#define Fifo boost::lockfree::spsc_queue<uchar4>
#endif

class FrameProvider
    {
    public:
	FrameProvider(uint w, uint h, string videoName, Version version);
	virtual ~FrameProvider(void);

	uchar4* getFrame();

    private:
	uchar4* loadFrame();
	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:
	//Input
	string nameVideo;
	uint w;
	uint h;

	Mat matRGBA;
	CVCaptureVideo capture;
	Version version;

	// full video
	const static int FRAME_NUMBER = 800; //51200
	uchar4* fullVideo[FRAME_NUMBER];
	int currentFrame = 0;
	bool videoLoaded = false;
	Fifo fifo();

    };
