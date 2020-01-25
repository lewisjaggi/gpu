#pragma once
#include <boost/lockfree/spsc_queue.hpp>
#include "cudaTools.h"

class FrameProvider;

#define Fifo boost::lockfree::spsc_queue<uchar4*, boost::lockfree::capacity<1024> >

class MyRunnable
    {
    public:
	MyRunnable();
    	virtual ~MyRunnable(void);
    	void run(void);
    	void init(Fifo* , FrameProvider*);


    	Fifo* fifo = nullptr;
	FrameProvider* frameProvider = nullptr;
    };
