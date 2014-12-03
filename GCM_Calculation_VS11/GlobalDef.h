#define featureDim 334
#define subSpaceDim 10
#define maxFrameNum 334


void deleteMatrix(double** matrix, int dimension);


double** newMatrix(int m, int n);


void subMatrix(double** fullMatrix, double** subMatrix, int ibegin, int ilength, int jbegin, int jlength);

