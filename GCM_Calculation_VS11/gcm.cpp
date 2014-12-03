#include "stdafx.h"
#include "gcm.h"


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
}


gcm::~gcm(void)
{
	//Release
	deleteMatrix(feature_ori, maxFrameNum);
	deleteMatrix(gcm_subspace, featureDim);
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


void gcm::deleteMatrix(double** matrix, int dimension)
{
	for (int i=0; i<dimension; i++)
	{
		delete matrix[i];
	}
	delete matrix;
}
