#include <iostream>
#include "MyRunnable.h"
#include "FrameProvider.h"

using std::cout;
using std::endl;
using std::string;
using std::to_string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
MyRunnable::MyRunnable()
    {

    }
MyRunnable::~MyRunnable()
    {
    }
/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/


/*--------------------------------------*\
 |*		Private		 	*|
 \*-------------------------------------*/


/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void MyRunnable::init(Fifo* fifo, FrameProvider* frameProvider)
    {
    this->fifo = fifo;
    this->frameProvider = frameProvider;
    }
void MyRunnable::run(void)
    {
    while(true)
 	{
 	uchar4* newFrame = frameProvider->loadFrame();
 	fifo->push(newFrame);
 	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/


/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
