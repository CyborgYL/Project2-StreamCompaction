
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


void scanNaive(const int *in, int *out, int len);
void scanSharedMem_singleblock(const int *in, int *out, int len);
void printArray(const int *out, const int len)
{
	for (size_t i = 0; i < len; i++)
	{
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}
__global__ void scanKernel(const int *in, int *out, int len)
{
	int thid = threadIdx.x;
	//out[thid] = (thid > 0) ? in[thid - 1] : 0;
	out[thid] = in[thid];
	__syncthreads();
	for (size_t offset = 1; offset < len; offset *= 2)
	{
		if (thid >= offset)
			out[thid] += out[thid - offset];
		__syncthreads();
	}
}
__global__ void scanKernel_shared_singleblock(const int *in, int *out, int len)
{
	extern __shared__ int temp[];
	int thid = threadIdx.x;
	int pin = 0, pout = 1;
	temp[pout*len + thid] = (thid > 0) ? in[thid - 1]: 0;
	__syncthreads();
	for (size_t offset = 1; offset < len; offset *= 2)
	{
		pout = 1 - pout;
		pin = 1 - pout;
		if (thid >= offset)
			temp[pout*len + thid] += temp[pin*len + thid - offset];
		else
		{
			temp[pout * len + thid] = temp[pin*len + thid];
		}
		__syncthreads();
	}
	out[thid] = temp[pout * len + thid];
}

int main()
{
	const int len = 10;
	int in[] = { 0, 0, 3, 4, 0, 6, 6, 7, 0, 7 };
	int *out = new int[len];
	for (size_t i = 0; i < len; i++)
	{
		out[i] = 0; 
	}
	scanNaive(in, out, len);
	printArray(out, len);

	scanSharedMem_singleblock(in, out, len);
	printArray(out, len);
	delete[] out;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void scanNaive(const int *in, int *out, int len)
{
	int *dev_in = 0;
	int *dev_out = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemCpy failed!1");
	}
	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemCpy failed!2");
	}
	scanKernel << <1, len >> >(dev_in, dev_out, len);
	
	cudaStatus = cudaMemcpy(out, dev_out, len * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemCpy failed!3");
	}
}

void scanSharedMem_singleblock(const int *in, int *out, int len)
{
	int *dev_in = 0;
	int *dev_out = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemCpy failed!1");
	}
	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemCpy failed!2");
	}
	scanKernel_shared_singleblock << <1, len, len * 2 * sizeof(int) >> >(dev_in, dev_out, len);

	cudaStatus = cudaMemcpy(out, dev_out, len * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemCpy failed!3");
	}
}