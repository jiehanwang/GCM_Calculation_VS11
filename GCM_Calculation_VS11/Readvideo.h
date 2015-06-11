//#include "PostureRecognition.h"
#include <time.h>
#include <opencv2\opencv.hpp>
#include<iostream>
#include<fstream>
#include "GlobalDef.h"

using namespace std;
using namespace cv;

#define r_width 640
#define r_heigth 480


class Readvideo
{
public:
	vector<LONGLONG> vDepthFrame;
	vector<LONGLONG> vColorFrame;
	vector<LONGLONG> vSkeletonFrame;
	vector<SLR_ST_Skeleton> vSkeletonData;		//一个手语词汇的骨架点集合
	vector<Mat> vDepthData;
	vector<IplImage*> vColorData;
public:
	void readvideo(string filePath);
	bool readColorFrame(string filename);
	bool readDepthFrame(string filename);
	bool readSkeletonFrame(string filename);
	Mat retrieveColorDepth(Mat depthMat);
	Readvideo(void);
	~Readvideo(void);
};

