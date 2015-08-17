#include "stdafx.h"
#include "gcm.h"
#include <fstream>

gcm::gcm(void)
{
	//Initial
	feature_ori = new double *[maxFrameNum];
	for (int i=0; i<maxFrameNum; i++)
	{
		feature_ori[i] = new double[featureDim];
	}

	gcm_subspace = new double*[featureDim];
	for (int i=0; i<featureDim; i++)
	{
		gcm_subspace[i] = new double[subspaceDim];
	}

	myModel = svm_load_model("..\\model\\model_UI4_noHandSeg_noPSVM");         //SVM model
	myModel_candi = svm_load_model("..\\model\\model_UI4_noHandSeg_PSVM");         //SVM model

	subFeaAll_model = newMatrix(featureDim, NClass*subSpaceDim*NTrainSample);  //Training Matrix
	fstream infile("..\\model\\subFeaAll_UI4_noHandSeg_noPSVM.dat",ios::in|ios::binary);
	for (int i=0; i<featureDim; i++)
	{
		for(int j=0;j<NClass*subSpaceDim*NTrainSample;j++)
		{
			infile.read((char*)&subFeaAll_model[i][j],sizeof(subFeaAll_model[i][j]));
		}
	}
	infile.close( );

	x = new svm_node[NClass*NTrainSample+1+1];                 //To release
	votewhj = new int[NClass*NTrainSample];               //To release

	nFrames = 0;
	nDimension = featureDim;

	prob_estimates = new double[NClass];

	subFea1 = newMatrix(featureDim, subSpaceDim);

	imgShow = cvCreateImage(cvSize(640, 480), 8,3);

	handSegmentVideo.init();
}


gcm::~gcm(void)
{
	//Release
	deleteMatrix(feature_ori, maxFrameNum);
	deleteMatrix(gcm_subspace, featureDim);
	deleteMatrix(subFeaAll_model,featureDim);

	//Destroy the SVM model
	svm_free_and_destroy_model(&myModel);

	delete []prob_estimates;

	deleteMatrix(subFea1,featureDim);

	cvReleaseImage(&imgShow);
}


void gcm::readInData(CString FileName)
{
	FILE* fp;
	char str[100];

	if ((fp = fopen(FileName, "r")) == NULL)
	{
		sprintf_s(str,"%s%s", "Invalid source test_filename, return.", "die"); 
		//AfxMessageBox(str);
		cout<<str<<endl;
		fclose(fp);
		return;
	}

	fscanf(fp, "%d%d", &nFrames, &nDimension);

	if (fseek(fp, 0L, SEEK_SET)) {
		//AfxMessageBox("file operation error, return");
		cout<<"file operation error, return"<<endl;
		fclose(fp);
		return;
	}

	GetData(fp, nFrames, &nFrames, feature_ori);
	fclose(fp);
}

double** gcm::GetData(FILE* fp, int Tmax, int *tl, double** data)
{
	int no_of_inputs,ip_dimension,il;
	float  buffer;
	//double **totaldata;

	fscanf(fp, "%d", &no_of_inputs);
	fscanf(fp, "%d", &ip_dimension);

	*tl=no_of_inputs;

	//totaldata=(double**)Alloc2d(no_of_inputs,ip_dimension,sizeof(double));
	for (int iin = 0; iin < no_of_inputs; iin++) 
	{
		for ( il = 0; il < ip_dimension; il++)	
		{
		  
	    	if (fscanf(fp, "%f", &buffer) == EOF)
				exit(1);
			data[iin][il] = buffer;
		}
	}
	//return (totaldata);
	return NULL;
}

char** gcm::Alloc2d(int dim1, int dim2,int size)
{
	int		i;
	unsigned	nelem;
	char	*p, **pp;

	nelem = (unsigned) dim1*dim2;

	p = (char *)calloc(nelem, (unsigned) size);

	if( p == NULL ) {
		return(NULL);
	}

	pp = (char **) calloc((unsigned) dim1, (unsigned) sizeof(char *));

	if (pp == NULL)
	{
		free(p);
		return( NULL );
	}

	for(int i=0; i<dim1; i++)
		pp[i] = p + i*dim2*size; 

	return(pp);	
}

void gcm::gcmSubspace(void)
{
	int n = nFrames;
	//white
	for (int d=0; d<nDimension; d++)
	{
		double dimAve = 0.0;
		for (int f=0; f<nFrames; f++)
		{
			dimAve += feature_ori[f][d];
		}
		dimAve /= nFrames;
		for (int f=0; f<nFrames; f++)
		{
			feature_ori[f][d] -= dimAve;
		}
	}

	//Obtain the Covariance
	double** C;
	C = new double*[nDimension];
	for (int i=0; i<nDimension; i++)
	{
		C[i] = new double[nDimension];
	}


	for (int d=0; d<nDimension; d++)
	{
		for (int d2=0; d2<nDimension; d2++)
		{
			C[d][d2] = 0.0;
			for (int f=0; f<nFrames; f++)
			{
				C[d][d2] += feature_ori[f][d]*feature_ori[f][d2];
			}
		}
	}

	for (int d=0; d<nDimension; d++)
	{
		for (int d2=0; d2<nDimension; d2++)
		{
			C[d][d2] /= nFrames;
			if (d == d2)
			{
				C[d][d2] += 0.001;
			}
		}
	}


	//SVD to obtain the subspace
	double* w;
	w = new double[nDimension];
	double **v;
	v = new double*[nDimension];
	for (int i=0; i<nDimension; i++)
	{
		v[i] = new double[nDimension];
	}

	gcmSVD.svdcmp(C, nDimension, nDimension, w, v);

	//The C[all][reDim =10] is the subspace
	for (int i=0; i<nDimension; i++)
	{
		for (int j=0; j<subspaceDim; j++)
		{
			gcm_subspace[i][j] = C[i][j];
		}
	}

	//Release
	deleteMatrix(C,nDimension);
	delete w;
	deleteMatrix(v, nDimension);
}


double** gcm::GenerateSubspace(CString FileName)
{
	readInData(FileName);
	gcmSubspace();
	return gcm_subspace;
}

double** gcm::GetSubspace()
{
	return gcm_subspace;
}


void gcm::deleteMatrix(double** matrix, int dimension)
{
	for (int i=0; i<dimension; i++)
	{
		delete matrix[i];
	}
	delete matrix;
}


void gcm::oriData2Feature(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, vector<IplImage*> vColorData)
{
	nDimension = featureDim;
	//The frame size
	int frameSize = vColorData.size();
	//Obtain the limited height
	int heightLimit = min(vSkeletonData[0]._2dPoint[7].y, vSkeletonData[0]._2dPoint[11].y) - 20;
	//Compute the keyframe mask
// 	KeyFrame myKeyFrame;
// 	int* KeyFrameLabel;
// 	KeyFrameLabel = new int[frameSize];
// 	myKeyFrame.getKeyFrameLabel(vSkeletonData,15,heightLimit,KeyFrameLabel);

	//To get the useful frame numbers. The key frame judge can be in here.
//	int framSize_Real = 0;
//	int* frameMask = new int[frameSize];
//	for (int i=0; i<frameSize; i++)
//	{
//		int heightThisLimit = min(vSkeletonData[i]._2dPoint[7].y, vSkeletonData[i]._2dPoint[11].y);
//		if (heightThisLimit < heightLimit /*&& KeyFrameLabel[i] == 1*/)
//		{
//			framSize_Real++;
//			frameMask[i] = 1;
//		}
//		else
//		{
//			frameMask[i] = 0;
//		}
//	}

	//Compute the features
	int totalFrameNum = 0;
	
	CvPoint headPoint, lPoint2, rPoint2;
	//system("rd /s/q ..\\output\\handImages" );
	//mkdir("..\\output\\handImages");
	
	
	
#ifdef SHOWVIDEO
	cvNamedWindow("video",1);
	
#endif
	for (int i=0; i<frameSize; i++)
	{
			//Face detection, executed only once. 
		if (i == 0)
		{
			headPoint.x = vSkeletonData[i]._2dPoint[3].x;
			headPoint.y = vSkeletonData[i]._2dPoint[3].y;

			bool bHeadFound = handSegmentVideo.headDetectionVIPLSDK(vColorData[i],
				vDepthData[i],
				headPoint);

			if(bHeadFound)
			{
				handSegmentVideo.colorClusterCv(handSegmentVideo.m_pHeadImage,3);
				handSegmentVideo.getFaceNeckRegion(vColorData[i],vDepthData[i]);
				handSegmentVideo.copyDepthMat(vDepthData[i].clone());
			}
		}

		int heightThisLimit = min(vSkeletonData[i]._2dPoint[7].y,
			vSkeletonData[i]._2dPoint[11].y);
		if (heightThisLimit<heightLimit /*&& KeyFrameLabel[i] == 1*/)
		{
			//////////////////////////////////////////////////////////////////////////
			//Color Image
			Posture posture;
					
			lPoint2.x = vSkeletonData[i]._2dPoint[7].x;
			lPoint2.y = vSkeletonData[i]._2dPoint[7].y;

			rPoint2.x = vSkeletonData[i]._2dPoint[11].x;
			rPoint2.y = vSkeletonData[i]._2dPoint[11].y;
					
			CvRect leftHand;
			CvRect rightHand;
					
			handSegmentVideo.kickHandsAll(vColorData[i],vDepthData[i]
			,lPoint2,rPoint2,posture,leftHand,rightHand);

			if(rightHand.x<0 || rightHand.y<0 || rightHand.height<0 || rightHand.width<0)
			{
				rightHand.x = 0;
				rightHand.y = 0;
				rightHand.height = 10;
				rightHand.width = 10;
			}
			if(leftHand.x<0 || leftHand.y<0 || leftHand.height<0 || leftHand.width<0)
			{
				leftHand.x = 0;
				leftHand.y = 0;
				leftHand.height = 10;
				leftHand.width = 10;
			}

			
			//////////////////////////////////////////////////////////////////////////
			//Tskp
			Vector<float> skP;
			for (int sk=3; sk<13; sk+=2)
			{
				for (int sk2=sk+2; sk2<13; sk2+=2)
				{
					float temp = pow((vSkeletonData[i]._3dPoint[sk].x - vSkeletonData[i]._3dPoint[sk2].x),2)
						+ pow((vSkeletonData[i]._3dPoint[sk].y - vSkeletonData[i]._3dPoint[sk2].y),2)
						+ pow((vSkeletonData[i]._3dPoint[sk].z - vSkeletonData[i]._3dPoint[sk2].z),2);
					skP.push_back(temp);
				}
			}

			float maxSKP = 0.0;
			for (int k=0; k<skP.size(); k++)
			{
				if (maxSKP < skP[k])
				{
					maxSKP = skP[k];
				}
			}
			for (int k=0; k<skP.size(); k++)
			{
				skP[k] /=maxSKP;
				feature_ori[totalFrameNum][k] = skP[k];
			}
			//cout<<endl;
			/////////////////////////////////////////////////////////////////////////
			//HoG
			double hog_color[DES_FEA_NUM];
			handSegmentVideo.getHogFeature(posture.leftHandImg, posture.rightHandImg,hog_color);

			for (int d=0; d<DES_FEA_NUM; d++)
			{
				//cout<<hog_color[d]<<" ";
				feature_ori[totalFrameNum][d+10] = hog_color[d];
			}
			//cout<<endl;

#ifdef SHOWVIDEO
			//For Debug
			//cvRectangle(vColorData[i], cvPoint((handSegmentVideo.m_faceRect).x, (handSegmentVideo.m_faceRect).y),
			//	cvPoint((handSegmentVideo.m_faceRect).x+(handSegmentVideo.m_faceRect).width, headPoint.y+headPoint.height), cvScalar(0,0,255),3,8,0);
			
			cvCopy(vColorData[i], imgShow, NULL);
			cvRectangle(imgShow, cvPoint(leftHand.x, leftHand.y),
				cvPoint(leftHand.x+leftHand.width, leftHand.y+leftHand.height), cvScalar(0,0,255),2,8,0);
			cvRectangle(imgShow, cvPoint(rightHand.x, rightHand.y),
				cvPoint(rightHand.x+rightHand.width, rightHand.y+rightHand.height), cvScalar(0,0,255),2,8,0);
#endif		


			//Save hand images. 
			CString handImageFileName;
			handImageFileName.Format("..\\output\\handImages\\left_%03d.jpg",totalFrameNum);
			cvSaveImage(handImageFileName, posture.leftHandImg);
			handImageFileName.Format("..\\output\\handImages\\right_%03d.jpg",totalFrameNum);
			cvSaveImage(handImageFileName, posture.rightHandImg);

			totalFrameNum++;
		}
		else
		{
#ifdef SHOWVIDEO
			cvCopy(vColorData[i], imgShow, NULL);
#endif
	    }
#ifdef SHOWVIDEO
		cvShowImage("video",imgShow);
		cvWaitKey(20);		
#endif
		
	}
	
#ifdef SHOWVIDEO
	cvDestroyWindow("video");
#endif
	nFrames = totalFrameNum;
	//cout<<"totalFrameNum "<<totalFrameNum<<endl;

}


int gcm::patchRun(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, vector<IplImage*> vColorData, 
				  int *rankIndex, double *rankScore)
{
	int kernelFeatureDim = NClass*NTrainSample;
	//clock_t startT=clock();
	oriData2Feature(vSkeletonData, vDepthData, vColorData);
	//cout<<"=======Time========="<<clock()-startT<<endl;
 	gcmSubspace();

	x[0].index = 0;
	for (int j=0; j<kernelFeatureDim; j++)
	{
		subMatrix(subFeaAll_model, subFea1, 0, featureDim, j*subSpaceDim, subSpaceDim);
		x[j+1].value = myGcmKernel.Frobenius(subFea1, gcm_subspace, featureDim, subSpaceDim);
		x[j+1].index=j+1;
	}
	x[kernelFeatureDim+1].index=-1;

	//int testID = svm_predict_probability(myModel, x, prob_estimates);
	int testID_noPro = svm_predict(myModel, x);

	int testID = svm_predict_probability(myModel_candi, x, prob_estimates);

	//Sort and get the former 5 ranks. 
	vector<scoreAndIndex> rank;
	for (int i=0; i<myModel->nr_class; i++)
	{
		scoreAndIndex temp;
		temp.index = myModel->label[i];
		temp.score = prob_estimates[i];
		rank.push_back(temp);
	}
	sort(rank.begin(),rank.end(),comp);

	
	rankIndex[0] = testID_noPro;
	rankScore[0] = 1.0;
	int candiN = 0;
	//for (int i=1; i<5; i++)
	int seqCandiN = 1;
	while(seqCandiN<5)
	{
		if (rank[candiN].index == testID_noPro)
		{
			candiN++;
			continue;
		}
		rankIndex[seqCandiN] = rank[candiN].index;
		rankScore[seqCandiN] = rank[candiN].score;
		candiN++;
		seqCandiN++;
	}
	releaseResource();
	return rankIndex[0];
}



void gcm::releaseResource(void)
{
	nFrames = 0;
}
bool gcm::comp(scoreAndIndex dis_1, scoreAndIndex dis_2)
{
	return dis_1.score > dis_2.score;
}

void gcm::computePQ(vector<double*> P, vector<double**> Q, int nFrames_PQ)
{
	//The loop for all the frames. 
	for (int i=0; i<nFrames_PQ; i++)
	{
		//Compute P
		double* tempP = new double[featureDim];
		for (int j=0; j<featureDim; j++)
		{
			tempP[j] = 0;
			for (int k=0; k<i; k++)
			{
				tempP[j] += feature_ori[k][j];
			}
		}
		P.push_back(tempP);    //To release it when reaching the end of a sentence 

		//Compute Q
		double** tempQ;
		tempQ = new double *[featureDim];
		for (int f=0; f<featureDim; f++)
		{
			tempQ[f] = new double[featureDim];
		}
		for (int f1=0; f1<featureDim; f1++)
		{
			for (int f2=0; f2<featureDim; f2++) 
			{
				tempQ[f1][f2] = 0;
				for (int l=0; l<i; l++)
				{
					tempQ[f1][f2] += feature_ori[f1][l]*feature_ori[f2][l];
				}
			}
		}
		Q.push_back(tempQ);      //To release it when reaching the end of a sentence 
	}
}

//This function is used to recognize continuous SLR in a off-line way. 
//Since the data should be read in all at once, the class gcmCont is not necessary used. 
int gcm::patchRun_continuous_PQ(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, vector<IplImage*> vColorData, 
								int *rankIndex, double *rankScore)
{
	int window = 40;
	int kernelFeatureDim = NClass*NTrainSample;
	//Computing features
	cout<<"Computing features..."<<endl;
	oriData2Feature(vSkeletonData, vDepthData, vColorData);

	//////////////////////////////////////////////////////////////////////////
	//Compute P and Q all at once.
	vector<double*> P;
	vector<double**> Q;
	cout<<"Computing P and Q..."<<endl;

	//computePQ(P, Q, vColorData.size());
	//white
	for (int d=0; d<nDimension; d++)
	{
		double dimAve = 0.0;
		for (int f=0; f<nFrames; f++)
		{
			dimAve += feature_ori[f][d];
		}
		dimAve /= nFrames;
		for (int f=0; f<nFrames; f++)
		{
			feature_ori[f][d] -= dimAve;
		}
	}

	//Compute P and Q.
	int nFrames_PQ = vColorData.size();
	for (int i=0; i<nFrames_PQ; i++)
	{
		cout<<"Current "<<i<<"/"<<nFrames_PQ<<endl;
		//Compute P
		double* tempP = new double[featureDim];
		for (int j=0; j<featureDim; j++)
		{
			tempP[j] = 0;
			for (int k=0; k<i; k++)
			{
				tempP[j] += feature_ori[k][j];
			}
		}
		P.push_back(tempP);    //To release it when reaching the end of a sentence 

		//Compute Q
		double** tempQ;
		tempQ = newMatrix(featureDim, featureDim);
		for (int f1=0; f1<featureDim; f1++)
		{
			for (int f2=0; f2<featureDim; f2++) 
			{
				tempQ[f1][f2] = 0;
				for (int l=0; l<i; l++)
				{
					tempQ[f1][f2] += feature_ori[f1][l]*feature_ori[f2][l];
				}
			}
		}
		Q.push_back(tempQ);      //To release it when reaching the end of a sentence 
	}
	//////////////////////////////////////////////////////////////////////////	
	
	double** C = newMatrix(featureDim, featureDim);
	double* p_temp = new double[featureDim];  //The deltaP
	double** Pm = newMatrix(featureDim, featureDim);
	for (int i=window; i<vColorData.size()-window; i++)
	{
		int begin = i-window/2;
		int end = i+window/2;

		//The matrix from p
		for (int pf=0; pf<featureDim; pf++)
		{
			p_temp[pf] = P[end][pf]-P[begin][pf];
		}
		vector2matrix(p_temp, p_temp, Pm,featureDim);

		//Compute the covariance matrix
		for (int l=0; l<featureDim; l++)
		{
			for (int m=0; m<featureDim; m++)
			{
				C[l][m] = ((Q[end][l][m]-Q[begin][l][m])-Pm[l][m]/(end-begin+1))/(end-begin);
			}
		}

		//Regularization term added
		for (int d=0; d<nDimension; d++)
		{
			for (int d2=0; d2<nDimension; d2++)
			{
				//C[d][d2] /= nFrames;
				if (d == d2)
				{
					C[d][d2] += 0.001;
				}
			}
		}

		//The subspace matrix
		PQsubspace(C, gcm_subspace);

		//For debug
		ofstream foutDebug;
		foutDebug.open("..\\output\\debug.txt");
		for (int i=0; i<featureDim; i++)
		{
			for (int j=0; j<subSpaceDim; j++)
			{
				foutDebug<<gcm_subspace[i][j]<<"\t";
			}
			foutDebug<<"\n";
		}
		foutDebug << flush;
		foutDebug.close();

		//The SVM classification
		x[0].index = 0;
		for (int j=0; j<kernelFeatureDim; j++)
		{
			subMatrix(subFeaAll_model, subFea1, 0, featureDim, j*subSpaceDim, subSpaceDim);
			x[j+1].value = myGcmKernel.Frobenius(subFea1, gcm_subspace, featureDim, subSpaceDim);
			x[j+1].index=j+1;
		}
		x[kernelFeatureDim+1].index=-1;

		int testID = svm_predict_probability(myModel, x, prob_estimates);
		cout<<"Frame: "<<i<<"/"<<vColorData.size()-window<<" Result: "<<testID<<endl;
	}
	delete[] p_temp;
	deleteMatrix(Pm,featureDim);
	deleteMatrix(C, featureDim);

	//To delete P and Q


	return 1;

// 	gcmSubspace();
// 
// 	x[0].index = 0;
// 	for (int j=0; j<kernelFeatureDim; j++)
// 	{
// 		subMatrix(subFeaAll_model, subFea1, 0, featureDim, j*subSpaceDim, subSpaceDim);
// 		x[j+1].value = myGcmKernel.Frobenius(subFea1, gcm_subspace, featureDim, subSpaceDim);
// 		x[j+1].index=j+1;
// 	}
// 	x[kernelFeatureDim+1].index=-1;
// 
// 	int testID = svm_predict_probability(myModel, x, prob_estimates);
// 
// 	//Sort and get the former 5 ranks. 
// 	vector<scoreAndIndex> rank;
// 	for (int i=0; i<myModel->nr_class; i++)
// 	{
// 		scoreAndIndex temp;
// 		temp.index = myModel->label[i];
// 		temp.score = prob_estimates[i];
// 		rank.push_back(temp);
// 	}
// 	sort(rank.begin(),rank.end(),comp);
// 
// 	for (int i=0; i<5; i++)
// 	{
// 		rankIndex[i] = rank[i].index;
// 		rankScore[i] = rank[i].score;
// 	}
// 
// 
// 	//Release
// 	nFrames = 0;
// 
// 	return testID;
}

void gcm::vector2matrix(double* v, double* u, double** m, int d)
{//v and u are two vectors. m is the output matrix. d is the length of the vector.
//v    u     m
//|   ---     
//|    
// 	m = new double*[d];
// 	for (int i=0; i<d; i++)
// 	{
// 		m[i] = new double[d];
// 	}
	//m = newMatrix(d,d);
	for (int i=0; i<d; i++)
	{
		for (int j=0; j<d; j++)
		{
			m[i][j] = v[i]*u[j];
		}
	}
}



void gcm::PQsubspace(double** C, double** PQ_subspace)
{
	//SVD to obtain the subspace
	double* w;
	w = new double[nDimension];
	double **v;
	v = newMatrix(nDimension,nDimension);

	gcmSVD.svdcmp(C, nDimension, nDimension, w, v);

	//The C[all][reDim =10] is the subspace
	for (int i=0; i<nDimension; i++)
	{
		for (int j=0; j<subspaceDim; j++)
		{
			PQ_subspace[i][j] = C[i][j];
		}
	}

	//Release
	//deleteMatrix(C,nDimension);
	delete w;
	deleteMatrix(v, nDimension);
}