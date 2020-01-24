#include <iostream>
#include "MyRunnable.h"


using std::cout;
using std::endl;
using std::string;
using std::to_string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
MyRunnable::MyRunnable(Fifo* fifo, FrameProvider* frameProvider)
    {
    this->fifo = fifo;
    this->frameProvider = frameProvider;
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
void MyRunnable::run(void)
    {
    cout<< "run" << endl;
    while(true)
 	{
	cout<< "run while"<< endl;
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