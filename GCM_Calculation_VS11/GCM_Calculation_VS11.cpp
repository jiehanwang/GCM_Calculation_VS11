// GCM_Calculation_VS11.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "gcm.h"


int _tmain(int argc, _TCHAR* argv[])
{
	gcm myGcm;
	CString fileName = "..\\input\\w0000.txt";
	double** subFea = myGcm.GenerateSubspace(fileName);
	return 0;
}

