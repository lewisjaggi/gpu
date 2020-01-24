#pragma once
#include "cpptest.h"

class TestConvolution: public Test::Suite
    {
	public:
	    TestConvolution();

	private:
	    void testConvolution_v1(void);
	    void testConvolution_v3_4(void);
    };
