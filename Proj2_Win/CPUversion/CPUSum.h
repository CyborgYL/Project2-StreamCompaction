#pragma once
class CPUSum
{
public:
	void scan_inclusive(const int *in, int *out,int n);
	void scan_exclusive(const int*in, int *out, int n);
	void transform_bool(const int *in, int *out, int n);
	void print(const int *a, int n);
	CPUSum();
	~CPUSum();
};

