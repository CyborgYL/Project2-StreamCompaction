// CPUversion.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CPUSum.h"

int _tmain(int argc, _TCHAR* argv[])
{
	const int len = 10;
	int a[] = { 0,0,3,4,0,6,6,7,0,7 };
	int *out = new int[len];
	int * out1 = new int[len];
	CPUSum CPUapp;
	CPUapp.transform_bool(a, out, len);
	CPUapp.print(out, len);
	CPUapp.scan_exclusive(out, out1, len);
	CPUapp.print(out1, len);

	delete[] out;
	return 0;
}

