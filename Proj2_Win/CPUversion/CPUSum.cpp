#include "stdafx.h"
#include "CPUSum.h"
#include <iostream>

CPUSum::CPUSum()
{
}


CPUSum::~CPUSum()
{
}

void CPUSum::scan_inclusive(const int *in, int *out, long n)
{
	out[0] = in[0];
	for (long i = 1; i < n; i++)
	{
		out[i] = out[i - 1] + in[i];
	}
}

void CPUSum::scan_exclusive(const int *in, int *out, long n)
{
	out[0] = 0;
	for (long i = 1; i < n; i++)
	{
		out[i] = out[i - 1] + in[i-1];
	}
}

void CPUSum::print(const int *a, long n)
{
	for (long i = 0; i < n; i++)
	{
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

void CPUSum::transform_bool(const int *in, int *out, long n)
{
	for (long i = 0; i < n; i++)
	{
		if (in[i] > 0)
			out[i] = 1;
		else
		{
			out[i] = 0;
		}
	}
}

void CPUSum::scatter(const int *in, int *out, const long len, long &newLength)
{
	int *temp = new int[len];
	int *temp2 = new int[len];
	transform_bool(in, temp, len);
	scan_inclusive(temp, temp2, len);
	newLength = temp2[len - 1];
	//out = new int[newLength];
	int pos = 0;
	if (in[0] > 0)
	{
		out[pos++] = in[0];
	}
	for (long i = 1; i < len; i++)
	{
		if (temp2[i] > temp2[i - 1])
		{
			out[pos++] = in[i];
		}
	}
	delete[] temp;
	delete[] temp2;
	//print(out, newLength);
	//delete[] out;
}