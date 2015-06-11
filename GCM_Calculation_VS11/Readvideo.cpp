#include "StdAfx.h"
#include "Readvideo.h"


Readvideo::Readvideo(void)
{
}


Readvideo::~Readvideo(void)
{
	vSkeletonData.clear();

	for(int i=0;i<vDepthData.size();i++)
		vDepthData[i].release();
	vDepthData.clear();

	for(int i=0;i<vColorData.size();i++)
		cvReleaseImage(&(vColorData[i]));
	vColorData.clear();
}

void Readvideo::readvideo(string filePath)
{
	clock_t start, durTime;
	start=clock();
	if( !( readColorFrame(filePath) && readDepthFrame(filePath) && readSkeletonFrame(filePath) ))
	{
		return;
	}	

	if( !( vSkeletonFrame.size() == vColorFrame.size() && vSkeletonFrame.size() == vDepthFrame.size() ))
	{
		return;
	}

	string str = filePath.substr(filePath.length()-25,21);
	cout<<str<<endl;
	cout<<"Reading..."<<endl;
	const char* savefilename=str.c_str();

	//读取彩色、深度和骨架数据
	const char* filePathChar=filePath.c_str();
	char filePathName[100];
	strcpy(filePathName,filePathChar);
	strcat(filePathName,"\\color.avi");
	CvCapture *capture = cvCreateFileCapture(filePathName);
	if( NULL == capture )
		return;

	int x,y;
	ifstream depthFileReader;
	depthFileReader.open(filePath+"\\depth.dat",ios::binary);
	if(depthFileReader == NULL)
		return;

	ifstream skeletonFileReader;
	skeletonFileReader.open(filePath+"\\skeleton.dat",ios::binary);
	if(skeletonFileReader == NULL)
		return;

	//read first color depth and skeleton data
	IplImage *frame = cvQueryFrame(capture);
	vColorData.push_back(cvCloneImage(frame));

	Mat depthMat;
	depthMat.create(r_heigth,r_width,CV_16UC1);
	ushort *depthData = new ushort[r_width*r_heigth];
	depthFileReader.read((char*)depthData,r_width*r_heigth*sizeof(ushort));
	if( !depthFileReader.fail() )
	{
		for(y=0; y<r_heigth; y++)
		{
			for(x=0; x<r_width; x++)
			{
				depthMat.at<ushort>(y,x) = depthData[y*r_width+x];
			}
		}
	}	
	delete [] depthData;
	vDepthData.push_back(depthMat.clone());

	SLR_ST_Skeleton mSkeleton;
	skeletonFileReader.read((char*)&mSkeleton,sizeof(mSkeleton));
	

	SLR_ST_Skeleton sLast = mSkeleton;
	SLR_ST_Skeleton sCurrent;
	int index = 0;

	vSkeletonData.clear();
	vSkeletonData.push_back(mSkeleton);
	
	int stillCount = 0;

	while(index < vSkeletonFrame.size()-1)
	{
		//读取下一帧的彩色、深度和骨架数据
		index++;
		frame = cvQueryFrame(capture);
		vColorData.push_back(cvCloneImage(frame));
		
		depthData = new ushort[r_width*r_heigth];
		depthFileReader.read((char*)depthData,r_width*r_heigth*sizeof(ushort));
		if( !depthFileReader.fail() )
		{
			for(y=0; y<r_heigth; y++)
			{
				for(x=0; x<r_width; x++)
				{
					depthMat.at<ushort>(y,x) = depthData[y*r_width+x];
				}
			}
		}	
		delete [] depthData;
		vDepthData.push_back(depthMat.clone());
		skeletonFileReader.read((char*)&sCurrent,sizeof(sCurrent));

		//if begin, judge end time and do hand sgementation
		vSkeletonData.push_back(sCurrent);
		sLast = sCurrent;

	}


	durTime=clock()-start;
	cout<<"Read Data Time:	"<<durTime<<endl;

	start=clock();

	depthFileReader.close();
	skeletonFileReader.close();
	cvReleaseCapture(&capture);
}

bool Readvideo::readColorFrame(string filename)
{
	vColorFrame.clear();
	ifstream colorFrameReader;
	const char* filePathChar=filename.c_str();
	char filePathName[100];
	strcpy(filePathName,filePathChar);
	strcat(filePathName,"\\color.frame");
	colorFrameReader.open(filePathName,ios::binary);
	if(colorFrameReader == NULL)
		return false;
	while( !colorFrameReader.eof() )
	{
		LONGLONG colorframeno;
		colorFrameReader.read((char*)&colorframeno,sizeof(LONGLONG));
		if( colorFrameReader.fail() )
			break;
		vColorFrame.push_back(colorframeno);
	}
	colorFrameReader.close();
	return true;
}

bool Readvideo::readDepthFrame(string filename)
{
	vDepthFrame.clear();
	ifstream depthFrameReader;
	const char* filePathChar=filename.c_str();
	char filePathName[100];
	strcpy(filePathName,filePathChar);
	strcat(filePathName,"\\depth.frame");
	depthFrameReader.open(filePathName,ios::binary);
	if(depthFrameReader == NULL)
		return false;
	while( !depthFrameReader.eof() )
	{
		LONGLONG depthframeno;
		depthFrameReader.read((char*)&depthframeno,sizeof(LONGLONG));
		if( depthFrameReader.fail() )
			break;
		vDepthFrame.push_back(depthframeno);
	}
	depthFrameReader.close();
	return true;
}

bool Readvideo::readSkeletonFrame(string filename)
{
	vSkeletonFrame.clear();
	ifstream skeletonFrameReader;
	const char* filePathChar=filename.c_str();
	char filePathName[100];
	strcpy(filePathName,filePathChar);
	strcat(filePathName,"\\skeleton.frame");
	skeletonFrameReader.open(filePathName,ios::binary);
	if(skeletonFrameReader == NULL)
		return false;
	while(!skeletonFrameReader.eof())
	{
		LONGLONG skeletonframeno;
		skeletonFrameReader.read((char*)&skeletonframeno,sizeof(LONGLONG));
		if( skeletonFrameReader.fail() )
			break;
		vSkeletonFrame.push_back(skeletonframeno);
	}
	skeletonFrameReader.close();
	return true;
}

Mat Readvideo::retrieveColorDepth(Mat depthMat)
{
	double maxDisp = -1.f;
	float S = 1.f;
	float V = 1.f;
	Mat disp;
	disp.create( Size(r_width,r_heigth), CV_32FC1);
	disp = cv::Scalar::all(0);
	for( int y = 0; y < disp.rows; y++ )
	{
		for( int x = 0; x < disp.cols; x++ )
		{
			unsigned short curDepth = depthMat.at<unsigned short>(y,x);
			if( curDepth != 0 )
				disp.at<float>(y,x) = (75.0 * 757) / curDepth;
		}
	}
	Mat gray;
	disp.convertTo( gray, CV_8UC1 );
	if( maxDisp <= 0 )
	{
		maxDisp = 0;
		minMaxLoc( gray, 0, &maxDisp );
	}
	Mat _depthColorImage;
	_depthColorImage.create( gray.size(), CV_8UC3 );
	_depthColorImage = Scalar::all(0);
	for( int y = 0; y < gray.rows; y++ )
	{
		for( int x = 0; x < gray.cols; x++ )
		{
			uchar d = gray.at<uchar>(y,x);
			if (d == 0)
				continue;

			unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

			unsigned int hi = (H/60) % 6;
			float f = H/60.f - H/60;
			float p = V * (1 - S);
			float q = V * (1 - f * S);
			float t = V * (1 - (1 - f) * S);

			Point3f res;

			if( hi == 0 ) //R = V,   G = t,   B = p
				res = Point3f( p, t, V );
			if( hi == 1 ) // R = q,   G = V,   B = p
				res = Point3f( p, V, q );
			if( hi == 2 ) // R = p,   G = V,   B = t
				res = Point3f( t, V, p );
			if( hi == 3 ) // R = p,   G = q,   B = V
				res = Point3f( V, q, p );
			if( hi == 4 ) // R = t,   G = p,   B = V
				res = Point3f( V, p, t );
			if( hi == 5 ) // R = V,   G = p,   B = q
				res = Point3f( q, p, V );

			uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
			uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
			uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

			_depthColorImage.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);     
		}
	}
	return _depthColorImage;
}