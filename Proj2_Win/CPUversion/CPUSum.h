#pragma once
class CPUSum
{
public:
	void scan_inclusive(const int *in, int *out,long n);
	void scan_exclusive(const int*in, int *out, long n);
	void transform_bool(const int *in, int *out, long n);
	void print(const int *a, long n);
	void scatter(const int *in, int *out, const long len, long &newLength);
	CPUSum();
	~CPUSum();
};

