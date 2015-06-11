#pragma once
#define SHOWVIDEO

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