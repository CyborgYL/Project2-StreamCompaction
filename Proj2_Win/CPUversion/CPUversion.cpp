// CPUversion.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CPUSum.h"
#include <time.h>
#define TEST_TIMES 100
#define LENGTH 100000
void test(const int *in, int *out,int len){
	CPUSum CPUapp;
	long newLen = len;
	CPUapp.scatter(in, out, len, newLen);
}
void warmup()
{
	clock_t start, finish;
	double duration;
	const int len = 1000;
	int newLen = len;
	int *in = new int[len];
	int *out = new int[len];
	int *out1 = new int[len];
	//int in[] = { 0, 0, 3, 4, 0, 6, 6, 7, 0, 7, 1, 3, 3, 2, 0, 1 };
	for (size_t i = 0; i < len; i++)
	{
		in[i] = i + 1;
	}
	int count = 10000;
	start = clock();
	while (count--)
	{
		test(in, out, len);
	}
	finish = clock();
	duration = finish - start;
	//CPUapp.print(out1, newLen);
	printf("warmup in %f ms\n", duration);

	delete[] out;
	delete[] out1;
}
int _tmain(int argc, _TCHAR* argv[])
{
	warmup();
	clock_t start, finish;
	double duration;
	const long len = LENGTH;
	int newLen = len;
	int *in = new int[len];
	int *out = new int[len];
	int *out1 = new int[len];
	//int in[] = { 0, 0, 3, 4, 0, 6, 6, 7, 0, 7, 1, 3, 3, 2, 0, 1 };
	for (long i = 0; i < len; i++)
	{
		in[i] = i + 1;
	}
	int count = TEST_TIMES;
	start = clock();
	while (count--)
	{
		test(in, out, len);
	}
	finish = clock();
	duration = finish - start;
	duration = duration / TEST_TIMES;
	//CPUapp.print(out1, newLen);
	printf("CPU Scattering finished in %f ms\n", duration);

	delete[] out;
	delete[] out1;
	return 0;
}

