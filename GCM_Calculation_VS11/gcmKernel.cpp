#include "StdAfx.h"
#include "gcmKernel.h"


gcmKernel::gcmKernel(void)
{
}


gcmKernel::~gcmKernel(void)
{
}


double gcmKernel::Frobenius(double** A, double** B, int dim, int n)
{
	//The definition of Grassmann manifold: A set of n-dimensional linear subspaces of the R^(dim).
	//n is the dimension of the subspace
	//dim is the dimension of the feature.  Frobenius

	
	double** C;
	C = new double*[n];
	for (int i=0; i<n; i++)
	{
		C[i] = new double[n];
	}

	double C_sum = 0.0;

	for (int i=0; i<n; i++)
	{
		for (int j=0; j<n; j++)
		{
			C[i][j] = 0.0;
			for (int d=0; d<dim; d++)
			{
				C[i][j] += (A[d][i]*B[d][j]);
				C_sum += (A[d][i]*B[d][j]);
			}
		}
	}


	C_sum = pow(C_sum, 0.5);
	return C_sum;
}
