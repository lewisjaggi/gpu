#include "TestMinMax.h"

#include "Device.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern bool isMinMax0To255Ok();
extern bool isMinMax23To177Ok();
extern bool isMinMaxLoopOk();
extern bool isMinMaxUniColorOk();

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Constructor		*|
 \*-------------------------------------*/

TestMinMax::TestMinMax()
    {
    TEST_ADD(TestMinMax::minMax0To255Ok);
    TEST_ADD(TestMinMax::minMax23To177Ok);
    TEST_ADD(TestMinMax::minMaxLoopOk);
    TEST_ADD(TestMinMax::minMaxUniColorOk);
    }

/*--------------------------------------*\
 |*		Methodes		*|
 \*-------------------------------------*/

void TestMinMax::minMax23To177Ok(void)
    {
    TEST_ASSERT(isMinMax23To177Ok());
    }

void TestMinMax::minMax0To255Ok(void)
    {
    TEST_ASSERT(isMinMax0To255Ok());
    }

void TestMinMax::minMaxLoopOk(void)
    {
    TEST_ASSERT(isMinMaxLoopOk());
    }

void TestMinMax::minMaxUniColorOk(void)
    {
    TEST_ASSERT(isMinMaxUniColorOk());
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

