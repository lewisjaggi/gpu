#pragma once
#include <boost/lockfree/spsc_queue.hpp>

class MyRunnable
    {
    public:
	MyRunnable(Fifo* fifo, FrameProvider* frameProvider);
    	virtual ~MyRunnable(void);
    	void run(void);


    	Fifo* fifo;
	FrameProvider* frameProvider;
    };
