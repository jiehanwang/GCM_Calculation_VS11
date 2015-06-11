#pragma once
#include "GlobalDef.h"
#include <vector>
#include "SVD.h"
//#include <string>
#include <atlstr.h>
#include <iostream>
#include <opencv2\opencv.hpp>
//#include "KeyFrame_label.h"
#include "HandSegment.h"
#include "S_svm.h"
#include "gcmKernel.h"
#include <direct.h>
using namespace std;
using namespace cv;



class gcm
{
public:
	gcm(void);
	~gcm(void);

	s_SVD gcmSVD;
	double** feature_ori;     //[nFrame][nDimension]
	double** gcm_subspace;
	int nFrames;
	int nDimension;
	static const int subspaceDim = subSpaceDim;

	vector<float> myPCA[OriHOGSize];  

	svm_model *myModel;
	double** subFeaAll_model;

	CHandSegment handSegmentVideo;
	svm_node* x;
	int* votewhj;

	double *prob_estimates;                     //This can be used for continuous SLR

	double** subFea1;

	gcmKernel myGcmKernel;

	IplImage* imgShow;

	//vector<float> feature_ori;
	void readInData(CString FileName);
	double** GetData(FILE* fp, int Tmax, int *tl, double** data);
	char **Alloc2d(int dim1, int dim2,int size);
	void gcmSubspace(void);
	double** GenerateSubspace(CString FileName);
	void deleteMatrix(double** matrix, int dimension);
	void oriData2Feature(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, vector<IplImage*> vColorData);
	double** GetSubspace();
	int patchRun(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, vector<IplImage*> vColorData, 
		int *rankIndex, double *rankScore);
	void releaseResource(void);
	static bool comp(scoreAndIndex dis_1, scoreAndIndex dis_2);
};

