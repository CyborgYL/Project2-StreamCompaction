#include "stdafx.h"
#include "CPUSum.h"
#include <iostream>

CPUSum::CPUSum()
{
}


CPUSum::~CPUSum()
{
}

void CPUSum::scan_inclusive(const int *in,int *out, int n)
{
	out[0] = in[0];
	for (size_t i = 1; i < n; i++)
	{
		out[i] = out[i - 1] + in[i];
	}
}

void CPUSum::scan_exclusive(const int *in, int *out, int n)
{
	out[0] = 0;
	for (size_t i = 1; i < n; i++)
	{
		out[i] = out[i - 1] + in[i-1];
	}
}

void CPUSum::print(const int *a, int n)
{
	for (size_t i = 0; i < n; i++)
	{
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

void CPUSum::transform_bool(const int *in, int *out, int n)
{
	for (size_t i = 0; i < n; i++)
	{
		if (in[i] > 0)
			out[i] = 1;
		else
		{
			out[i] = 0;
		}
	}
}