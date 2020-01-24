#pragma once

//#define KERNEL_CONVOLUTION_SIZE 9

class NoyauCreator
    {
	/*--------------------------------------*\
	|*	 	Constructor	 	*|
	 \*-------------------------------------*/

    public:
	NoyauCreator();
	virtual ~NoyauCreator(void);

	/*--------------------------------------*\
	|*		 Methodes	 	*|
	 \*-------------------------------------*/

    public:
	float* getTabNoyau();

    private:
	void create(void);
	void check(void);
	static double sum(float* tab, int n);

	/*--------------------------------------*\
	|*	 	Attributs	 	*|
	 \*-------------------------------------*/
    private:
	// Outputs
	float* tabNoyau;
	static const int W = 9;
    };
