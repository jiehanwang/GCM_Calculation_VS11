#pragma once
#include "GlobalDef.h"
#include <vector>
#include "SVD.h"
//#include <string>
#include <atlstr.h>
#include <iostream>
using namespace std;

class gcm
{
public:
	gcm(void);
	~gcm(void);

	SVD gcmSVD;
	double** feature_ori;
	double** gcm_subspace;
	int nFrames;
	int nDimension;
	static const int subspaceDim = subSpaceDim;


	//vector<float> feature_ori;
	void readInData(CString FileName);
	double** GetData(FILE* fp, int Tmax, int *tl, double** data);
	char **Alloc2d(int dim1, int dim2,int size);
	void gcmSubspace(void);
	double** GenerateSubspace(CString FileName);
	void deleteMatrix(double** matrix, int dimension);
};

