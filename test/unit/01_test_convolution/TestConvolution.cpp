#include "TestConvolution.h"

extern bool isKernelConvolutionV1_OK();
extern bool isKernelConvolutionV3_4_OK();

TestConvolution::TestConvolution()
    {
    TEST_ADD (TestConvolution::testConvolution_v1);
    TEST_ADD (TestConvolution::testConvolution_v3_4);
    }

void TestConvolution::testConvolution_v1(void)
    {
    TEST_ASSERT(isKernelConvolutionV1_OK());
    }

void TestConvolution::testConvolution_v3_4(void)
    {

    TEST_ASSERT(isKernelConvolutionV3_4_OK());
    }

