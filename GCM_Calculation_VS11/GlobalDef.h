#pragma once
#include <opencv2\opencv.hpp>
//#define SHOWVIDEO

typedef struct scoreAndIndex
{
	double score;
	int index;
}scoreAndIndex;

#define featureDim 334      //Dimension of feature
#define subSpaceDim 10      //Dimension of subSpace. 
#define maxFrameNum 334     //
#define OriHOGSize 648

#define NClass     370   //370
#define NTrainSample 4   //5 at most

//Release the memory of a matrix
void deleteMatrix(double** matrix, int dimension);
//Initial the memory of a matrix
double** newMatrix(int m, int n);
//Get the sub-matrix from a large one
void subMatrix(double** fullMatrix, double** subMatrix, int ibegin, int ilength, int jbegin, int jlength);

//////////////////////////////////////////////////////////////////////////
/// @author xu zhihao
/// @struct _Vector2i
/// @brief skeleton data transformed data
//////////////////////////////////////////////////////////////////////////
struct _Vector2i
{
	int x;
	int y;
};

//////////////////////////////////////////////////////////////////////////
/// @author xu zhihao
/// @struct SLR_ST_Skeleton
/// @brief skeleton data  real data
//////////////////////////////////////////////////////////////////////////
struct _Vector4f
{
	float x;
	float y;
	float z;
	float w;
};
//Structure of skeleton
struct SLR_ST_Skeleton
{
	_Vector4f _3dPoint[20];    ///< real point
	_Vector2i _2dPoint[20];    ///< pix in color image
}; 

//////////////////////////////////////////////////////////////////////////
//Come from HandSegment.h
//#define UsePCA
const int SRC_FEA_NUM = 324;//1764;//324;
#ifdef UsePCA
const int DES_FEA_NUM = 51;//51;49
#endif
#ifndef UsePCA
const int DES_FEA_NUM = 324;//1764;//324;//51;
#endif

const int IMG_SIZE = 64;
struct Posture
{
	IplImage *leftHandImg;    ///< left hand image
	IplImage *rightHandImg;   ///< right hand image
	CvPoint leftHandPt;       ///< left hand point
	CvPoint leftWristPt;      ///< left wrist point
	CvPoint rightHandPt;      ///< right hand point
	CvPoint rightWristPt;     ///< right wrist point

	Posture():leftHandImg(NULL),rightHandImg(NULL) {};
};

struct ColorModel
{
	double mean_cr;     ///< mean of cr
	double mean_cb;     ///< mean of cb
	double d_cr;        ///< variance of cr
	double d_cb;        ///< variance of cb
	ColorModel():mean_cr(0),mean_cb(0),d_cr(0),d_cb(0){};
};
//////////////////////////////////////////////////////////////////////////