// GCM_Calculation_VS11.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gcm.h"
#include "gcmKernel.h"
#include "GlobalDef.h"



int _tmain(int argc, _TCHAR* argv[])
{
	int nClass = 20;

	//Obtain the subspaces of all the samples.
	double** subFeaAll = newMatrix(featureDim, nClass*subSpaceDim);    //subFeaAll[dim][n*nClass]
	for (int i=0; i<nClass; i++)
	{
		cout<<"The "<<i<<" sample"<<endl;
		gcm myGcm;
		CString fileName; 
		fileName.Format("..\\input\\w%04d.txt", i);
		double** subFea = myGcm.GenerateSubspace(fileName);  //subFea[dim][n]

		for (int f=i*subSpaceDim; f<(i+1)*subSpaceDim; f++)
		{
			for (int d=0; d<featureDim; d++)
			{
				subFeaAll[d][f] = subFea[d][f-i*subSpaceDim];
			}
		}
		
	}

	//Compute the kernel matrix
	double** kernelMatrix = newMatrix(nClass, nClass);
	for (int i=0; i<nClass; i++)
	{
		cout<<"Kernel for "<<i<<" class"<<endl;
		for (int j=0; j<nClass; j++)
		{
			gcmKernel myGcmKernel;
			double** subFea1 = newMatrix(featureDim, subSpaceDim);
			double** subFea2 = newMatrix(featureDim, subSpaceDim);
			subMatrix(subFeaAll, subFea1, 0, featureDim, i*subSpaceDim, subSpaceDim);
			subMatrix(subFeaAll, subFea2, 0, featureDim, j*subSpaceDim, subSpaceDim);
			kernelMatrix[i][j] = myGcmKernel.Frobenius(subFea1, subFea2, featureDim, subSpaceDim);
			deleteMatrix(subFea1, featureDim);
			deleteMatrix(subFea2, featureDim); 
		}
	}

	//Release
	deleteMatrix(subFeaAll, featureDim);
	deleteMatrix(kernelMatrix, nClass);


// 	1. 定义model：
// 		svm_model*    m_CLSvmModel;
// 
// 	2. 载入model：
// 		m_CLSvmModel = svm_load_model("all_3.model");
// 
// 	3. probe的特征：
// 		svm_node node[33];
// 	for (i=0; i<32; i++)
// 	{
// 		node[i].index = i;
// 		node[i].value = m_rgCLFeature[i];
// 
// 	}
// 	node[32].index = -1;
// 
// 	4. 将node和model放入预测，返回类别：
// 		int     m_motionClassNo;      //motion label
// 	m_motionClassNo=(int)svm_predict(m_CLSvmModel,node);






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

