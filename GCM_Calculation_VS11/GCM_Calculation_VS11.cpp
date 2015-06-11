// GCM_Calculation_VS11.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gcm.h"
#include "GlobalDef.h"
#include "Readvideo.h"
#include <fstream>
#include <opencv2\opencv.hpp>
#include "gcmCont.h"
//#include <direct.h>
using namespace cv;

//#define  CompKernel
#define  LoadKernel

//#define TRAIN
#define TEST
//#define ContinuousTest
//#define BatchTEST

void trainModel(int nClass, int nTrainSample,int ifeatureDim, int isubSpaceDim)
{//The code is used for model construction.

	int kernelFeatureDim = nClass*nTrainSample;

	//The label may be different from the index!!!!!
	int label[NClass];
	for (int i=0; i<nClass; i++)
	{
		label[i] = i;
	}
 
#ifdef CompKernel
	//Obtain the subspaces of all the samples.
	//subFeaAll[ifeatureDim][nClass*isubSpaceDim*nTrainSample]
	double** subFeaAll = newMatrix(ifeatureDim, nClass*isubSpaceDim*nTrainSample);  
	int feaIndexAll = 0;
	CString fileName; 
	gcm myGcm;
	for (int i=0; i<nClass; i++)
	{
		cout<<"The "<<i<<"th Class ";
		for (int t=0; t<nTrainSample; t++)
		{
			cout<<".";
			fileName.Format("..\\input\\test_%d\\w%04d.txt", t+50, i);
			double** subFea = myGcm.GenerateSubspace(fileName);  //subFea[ifeatureDim][isubSpaceDim]
			int feaIndex = 0;
			for (int f=0; f<isubSpaceDim; f++)
			{
				for (int d=0; d<ifeatureDim; d++)
				{
					subFeaAll[d][feaIndexAll] = subFea[d][feaIndex];
				}
				feaIndexAll++;
				feaIndex++;
			}
		}
		cout<<endl;
	}
	//Write the kernel of training data.
	ofstream outfile("..\\model\\subFeaAll.dat",ios::binary);
	for (int i=0; i<ifeatureDim; i++)
	{
		for(int j=0;j<nClass*isubSpaceDim*nTrainSample;j++)
		{
			outfile.write((char*)&subFeaAll[i][j],sizeof(subFeaAll[i][j]));
		}
	}
	outfile.close( );

	

	//Compute the kernel matrix
	double** kernelMatrix = newMatrix(nClass*nTrainSample, nClass*nTrainSample);
	for (int i=0; i<nClass*nTrainSample; i++)
	{
		cout<<"Kernel for "<<i<<" row"<<endl;
		//double maxElement = 0.00001;
		for (int j=0; j<nClass*nTrainSample; j++)
		{
			gcmKernel myGcmKernel;
			double** subFea1 = newMatrix(ifeatureDim, isubSpaceDim);
			double** subFea2 = newMatrix(ifeatureDim, isubSpaceDim);
			subMatrix(subFeaAll, subFea1, 0, ifeatureDim, i*isubSpaceDim, isubSpaceDim);
			subMatrix(subFeaAll, subFea2, 0, ifeatureDim, j*isubSpaceDim, isubSpaceDim);
			kernelMatrix[i][j] = myGcmKernel.Frobenius(subFea1, subFea2, ifeatureDim, isubSpaceDim);
			deleteMatrix(subFea1, ifeatureDim);
			deleteMatrix(subFea2, ifeatureDim); 
		}
	}

	//Write the kernel of training data.
	ofstream outfile_kernel("..\\model\\kernel.dat",ios::binary);
	for (int i=0; i<nClass*nTrainSample; i++)
	{
		for(int j=0;j<nClass*nTrainSample;j++)
		{
			outfile_kernel.write((char*)&kernelMatrix[i][j],sizeof(kernelMatrix[i][j]));
		}
	}
	outfile_kernel.close( );
#endif

#ifdef LoadKernel
	cout<<"Loading kernel matrix..."<<endl;
	double** kernelMatrix = newMatrix(nClass*nTrainSample, nClass*nTrainSample);  
	fstream infile("..\\model\\kernel.dat",ios::in|ios::binary);
	for (int i=0; i<nClass*nTrainSample; i++)
	{
		for(int j=0;j<nClass*nTrainSample;j++)
		{
			infile.read((char*)&kernelMatrix[i][j],sizeof(kernelMatrix[i][j]));
		}
	}
	infile.close( );

	cout<<"Loading subFeaAll..."<<endl;
	double** subFeaAll = newMatrix(ifeatureDim, nClass*isubSpaceDim*nTrainSample);  
	fstream infileM("..\\model\\subFeaAll_2.dat",ios::in|ios::binary);
	for (int i=0; i<ifeatureDim; i++)
	{
		for(int j=0;j<nClass*isubSpaceDim*nTrainSample;j++)
		{
			infileM.read((char*)&subFeaAll[i][j],sizeof(subFeaAll[i][j]));
		}
	}
	infileM.close( );
#endif

	//////////////////////////////////////////////////////////////////////////
	//Training SVM model 
	//SVM settings
	svm_parameter myPara;
	myPara.svm_type = C_SVC;
	myPara.kernel_type = PRECOMPUTED;  //LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED /* kernel_type */
	myPara.degree = 3;
	myPara.gamma = 0.0000;
	myPara.coef0 = 0;
	myPara.nu = 0.5;
	myPara.cache_size = 100;
	myPara.C = 1;   //10
	myPara.eps = 0.001;    //1e-5
	myPara.p = 0.1;
	myPara.shrinking = 1; 
	myPara.probability = 1;
	myPara.nr_weight = 0;
	myPara.weight_label = NULL;
	myPara.weight = NULL;
	
	svm_problem  myProblem;
	myProblem.l = kernelFeatureDim;     //The number of samples for training. the value is nClass*nSamples
	myProblem.y = new double[myProblem.l];
	myProblem.x = new svm_node *[myProblem.l];   

	for(int i=0;i<myProblem.l;i++)
	{
		svm_node *x_space = new svm_node[kernelFeatureDim+1+1];  //The last one is for -1
		x_space[0].index = 0;
		x_space[0].value = i+1;
		for (int j=0;j<kernelFeatureDim;j++)
		{
			x_space[j+1].index=j+1;
			x_space[j+1].value=kernelMatrix[i][j];   //a sample one row. col is the feature
		}
		x_space[kernelFeatureDim+1].index=-1;
		myProblem.x[i] = x_space;

		int index = i/nTrainSample;
		myProblem.y[i]=label[index];
	}
	svm_model *myModel = svm_train(&myProblem, &myPara);   
	svm_save_model("..\\model\\model_2", myModel);
	//Release
	deleteMatrix(subFeaAll, ifeatureDim);
	deleteMatrix(kernelMatrix, nClass);
}

int _tmain(int argc, _TCHAR* argv[])
{
	//For debug
// 	ofstream foutDebug;
// 	foutDebug.open("..\\output\\debug.txt");
// 	for (int i=0; i<featureDim; i++)
// 	{
// 		for (int j=0; j<subSpaceDim; j++)
// 		{
// 			foutDebug<<subFeaAll[i][j]<<"\t";
// 		}
// 		foutDebug<<"\n";
// 	}
// 	foutDebug << flush;
// 	foutDebug.close();

	////////////////////////////////////////////////////
	//Train model
#ifdef TRAIN
	trainModel(NClass, NTrainSample,featureDim,subSpaceDim);
#endif

	///////////////////////////////////////////////////////////
	//Original video test
#ifdef TEST
	//The object of GCM class
	gcm myGcm;

	//Search a data folder, find the samples to be tested. 
	bool fileFindFlag;
	CFileFind fileFind;
	CString normFileName;
	normFileName.Format("E:\\isolatedDemoSign4test\\P54\\*.oni");
	//normFileName.Format("C:\\Users\\汉杰\\Desktop\\TestData\\*.oni");
	fileFindFlag = true;
	fileFindFlag = fileFind.FindFile(normFileName);
	int correct = 0;

	//The loop of testing
	int *rankIndex;
	rankIndex = new int[5];
	double *rankScore;
	rankScore = new double[5];

	while (fileFindFlag)
	{
		fileFindFlag = fileFind.FindNextFile();
		CString videoFilePath = fileFind.GetFilePath();
		CString videoFileName = fileFind.GetFileName();
		CString videoFileClass = videoFileName.Mid(4,4);
		int classNo = _ttoi(videoFileClass);
		cout<<classNo<<endl;

		Readvideo myReadVideo;
		string s = (LPCTSTR)videoFilePath;
		myReadVideo.readvideo(s);
		int frameSize = myReadVideo.vColorData.size();
		cout<<"Total frameSize "<<frameSize<<endl;
		
		int result = myGcm.patchRun(myReadVideo.vSkeletonData, myReadVideo.vDepthData, myReadVideo.vColorData, 
			rankIndex, rankScore);

		//Show the result
		for (int i=0; i<5; i++)
		{
			cout<<rankIndex[i]<<'\t'<<rankScore[i]<<endl;
		}

		cout<<videoFileName<<", result: "<<result;
		if (result == classNo)
		{
			cout<<"-----Correct";
			correct++;
		}
		else
		{
			cout<<"-------Wrong";
		}
		cout<<endl<<endl;
	}
	delete []rankIndex;
	delete []rankScore;
	float acc = (float)correct/370;
	cout<<"Accuracy: "<<acc<<endl;
	
#endif


#ifdef ContinuousTest
	CString videoFilePath = "..\\input\\P54_0001_1_0_20121002.oni";
	Readvideo myReadVideo;
	string s = (LPCTSTR)videoFilePath;
	myReadVideo.readvideo(s);
	int frameSize = myReadVideo.vColorData.size();
	cout<<"Total frameSize "<<frameSize<<endl;

	gcmCont myGcmCont;
	vector<SLR_ST_Skeleton> cSkeletonData;
	vector<Mat> cDepthData;
	vector<IplImage*> cColorData;
	for (int i=0; i<frameSize; i++)
	{
		myGcmCont.frameUpdate(myReadVideo.vSkeletonData[i], myReadVideo.vDepthData[i], myReadVideo.vColorData[i]);
		if (myGcmCont.dataReady == true)
		{
			myGcmCont.recogCont();
		}

	}
#endif
	//////////////////////////////////////////////////////////////////////////
	//Batch Test 
#ifdef BatchTEST
	int kernelFeatureDim = NClass*NTrainSample;
	//Load the model and training data for kernel construction.
	cout<<"Loading models..."<<endl;
	svm_model *myModel = svm_load_model("..\\model\\model_2");         //SVM model
	double** subFeaAll = newMatrix(featureDim, NClass*subSpaceDim*NTrainSample);  //Training Matrix
	fstream infile("..\\model\\subFeaAll_2.dat",ios::in|ios::binary);
	for (int i=0; i<featureDim; i++)
	{
		for(int j=0;j<NClass*subSpaceDim*NTrainSample;j++)
		{
			infile.read((char*)&subFeaAll[i][j],sizeof(subFeaAll[i][j]));
		}
	}
	infile.close( );

	//Obtain the subspaces of the TEST samples.
	int nCorrect = 0;
	gcm myGcm;
	gcmKernel myGcmKernel;
	CString fileName;
	double** subFea1 = newMatrix(featureDim, subSpaceDim);
	double** subFea = newMatrix(featureDim, subSpaceDim);    
	svm_node* x = new svm_node[kernelFeatureDim+1+1];           //To release
	int* votewhj = new int[kernelFeatureDim];               //To release
	for (int i=0; i<NClass; i++)
	{
		fileName.Format("..\\input\\test_54\\w%04d.txt", i);
		subFea = myGcm.GenerateSubspace(fileName);  //subFea[dim][n]
		x[0].index = 0;
		for (int j=0; j<kernelFeatureDim; j++)
		{
				subMatrix(subFeaAll, subFea1, 0, featureDim, j*subSpaceDim, subSpaceDim);
				x[j+1].value = myGcmKernel.Frobenius(subFea1, subFea, featureDim, subSpaceDim);
				x[j+1].index=j+1;
		}
		x[kernelFeatureDim+1].index=-1;

		//int testID = svm_predict(myModel, x, votewhj);
		int testID = svm_predict(myModel, x);

		cout<<i<<"th result "<<testID;
		if (testID == i)
		{
			nCorrect += 1;
			cout<<"----Correct"<<endl;
		}
		else
		{
			cout<<"----Wrong"<<endl;
		}
	}
	deleteMatrix(subFea1, featureDim);
	deleteMatrix(subFea, featureDim);
	deleteMatrix(subFeaAll, featureDim);

	float correctRate = (float)nCorrect/NClass;
	cout<<"correctRate "<<correctRate<<endl;
#endif

	cout<<"Done!"<<endl;
	getchar();
	return 0;
}

