#pragma once
#include "GlobalDef.h"
#include <math.h>


class gcmKernel
{
public:
	gcmKernel(void);
	~gcmKernel(void);
	double Frobenius(double** A, double** B, int dim, int n);
};

