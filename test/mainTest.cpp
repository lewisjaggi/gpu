#include <stdlib.h>
#include <iostream>
#include <string>

#include "cppTest+.h"
#include "unit/01_test_convolution/TestConvolution.h"
#include "unit/02_test_minmax/TestMinMax.h"
#include "unit/03_test_amplification/TestAmplification.h"

using std::string;
using std::cout;
using std::endl;

using Test::Suite;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static bool testALL();

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainTest();

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainTest()
    {
    bool isOk = testALL();

    cout << "\nisOK = " << isOk << endl;

    return isOk ? EXIT_SUCCESS : EXIT_FAILURE;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool testALL()
    {
    Suite testSuite;
    testSuite.add(std::unique_ptr < Suite > (new TestConvolution()));
    testSuite.add(std::unique_ptr < Suite > (new TestMinMax()));
    testSuite.add(std::unique_ptr < Suite > (new TestAmplification()));

    string output = "out/test";
    return runTestHtml(output, testSuite); // Attention: html create in ./workingDirectory/out
//    return runTestConsole(output, testSuite);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

