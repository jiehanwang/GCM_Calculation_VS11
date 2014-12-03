#include "stdafx.h"
#include "GlobalDef.h"

void deleteMatrix(double** matrix, int dimension)
{
	for (int i=0; i<dimension; i++)
	{
		delete matrix[i];
	}
	delete matrix;
}

double** newMatrix(int m, int n)
{
	double** matrix;
	matrix = new double*[m];
	for (int i=0; i<m; i++)
	{
		matrix[i] = new double[n];
	}
	return matrix;
}

void subMatrix(double** fullMatrix, double** subMatrix, int ibegin, int ilength, int jbegin, int jlength)
{
	for (int i=ibegin; i<ibegin+ilength; i++)
	{
		for (int j=jbegin; j<jbegin+jlength; j++)
		{
			subMatrix[i-ibegin][j-jbegin] = fullMatrix[i][j];
		}
	}
}