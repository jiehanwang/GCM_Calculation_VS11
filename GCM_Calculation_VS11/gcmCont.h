#pragma once
#include "gcm.h"

const int windowSize = 60;


class gcmCont
{

public:
	//To store the frames in the sliding window. 
	vector<SLR_ST_Skeleton> cSkeletonData;
	vector<Mat> cDepthData;
	vector<IplImage*> cColorData;
	bool dataReady;
	gcm myGcm;

public:
	gcmCont(void);
	~gcmCont(void);
	void frameUpdate(SLR_ST_Skeleton vSkeletonData, Mat vDepthData, IplImage* vColorData);
	void recogCont(void);
	void frameReadinAll(SLR_ST_Skeleton vSkeletonData, Mat vDepthData, IplImage* vColorData);
};

