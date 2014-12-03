// GCM_Calculation_VS11.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gcm.h"
#include "gcmKernel.h"


int _tmain(int argc, _TCHAR* argv[])
{
	gcm myGcm1;
	CString fileName1 = "..\\input\\w0000.txt";
	double** subFea1 = myGcm1.GenerateSubspace(fileName1);

	gcm myGcm2;
	CString fileName2 = "..\\input\\w0001.txt";
	double** subFea2 = myGcm2.GenerateSubspace(fileName2);


	gcmKernel myGcmKernel;
	myGcmKernel.Frobenius(subFea1, subFea2, featureDim, 10);

	//SVM training and test.
	//How to train kernel SVM in C++
// 	TrainKernel = kernel(trainData,[],testID);
// 	ValKernel = kernel(trainData,testData,testID);
// 
// 	TTrainKernel = [(1:trainN)',TrainKernel];
// 		VValKernel = [(1:testN)',ValKernel'];
// 
// 	model_precomputed = svmtrain(trainLabel, TTrainKernel, '-t 4');
// 	[predict_label_P1, accuracy_P1, dec_values_P1] = svmpredict(testLabel, VValKernel, model_precomputed);


	return 0;
}

