#include "StdAfx.h"
#include "gcmCont.h"


gcmCont::gcmCont(void)
{
	dataReady = false;
}


gcmCont::~gcmCont(void)
{
}


void gcmCont::frameUpdate(SLR_ST_Skeleton vSkeletonData, Mat vDepthData, IplImage* vColorData)
{
	if (cSkeletonData.size() < windowSize)
	{
		cSkeletonData.push_back(vSkeletonData);
		cDepthData.push_back(vDepthData);
		cColorData.push_back(vColorData);
		//dataReady = false;
	}
	else
	{
		cSkeletonData.erase(cSkeletonData.begin());
		cDepthData.erase(cDepthData.begin());
		cColorData.erase(cColorData.begin());

		cSkeletonData.push_back(vSkeletonData);
		cDepthData.push_back(vDepthData);
		cColorData.push_back(vColorData);
		//dataReady = true;
	}
	if (cSkeletonData.size() == windowSize)
	{
		dataReady = true;
	}
	else
	{
		dataReady = false;
	}
}


void gcmCont::recogCont(void)
{
	int *rankIndex;
	rankIndex = new int[5];
	double *rankScore;
	rankScore = new double[5];

	int result = myGcm.patchRun(cSkeletonData, cDepthData, cColorData, rankIndex, rankScore);
	cout<<result<<" "<<rankScore[0]<<endl;

	delete []rankIndex;
	delete []rankScore;


	
}
