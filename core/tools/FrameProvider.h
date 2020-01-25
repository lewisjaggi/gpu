#pragma once

#include <iostream>

#include <boost/lockfree/spsc_queue.hpp>
#include "cudaTools.h"
#include "CVCaptureVideo.h"
#include "Version.h"
#include "MyRunnable.h"



using cv::Mat;
using std::string;



#define Fifo boost::lockfree::spsc_queue<uchar4*, boost::lockfree::capacity<1024> >




class FrameProvider
    {
    public:
	FrameProvider(uint w, uint h, string videoName, Version version);
	virtual ~FrameProvider(void);

	uchar4* getFrame();
	void pushFrame();
	uchar4* loadFrame();
	void start();

    private:

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
	MyRunnable runnable;

	// full video
	const static int FRAME_NUMBER = 800; //51200
	uchar4* fullVideo[FRAME_NUMBER];
	int currentFrame = 0;
	bool videoLoaded = false;
	Fifo fifo;

    };
