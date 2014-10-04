
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <thrust/scan.h>

#define BLOCK_SIZE 512
#define TEST_TIMES 1
#define LENGTH 10000000

void scanNaive(const int *in, int *out, int len);
void scanSharedMem_singleblock(const int *in, int *out, int len);
void scanSharedMem_multiblock(const int *in, int *out, int len);
void scatter(int *in, int *out, int len);
void scatter_thrust(int *in, int *out, int len);
void printArray(const int *out, const int len)
{
	for (size_t i = 0; i < len; i++)
	{
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}
void clearArray(int *out, const int len)
{
	for (size_t i = 0; i < len; i++)
	{
		out[i] = 0;
	}
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
	temp[pin * len + thid] = in[thid];
	//temp[pout*len + thid] = (thid > 0) ? temp[pin * len + thid -1]: 0;
	temp[pout*len + thid] = temp[pin*len + thid];
	__syncthreads();
	for (size_t offset = 1; offset < len; offset *= 2)
	{
		if (thid >= offset)
			temp[pout * len + thid] += temp[pout * len + thid - offset];
		__syncthreads();
	}
	out[thid] = temp[pout * len + thid];
}

__device__ void scanKernel_singleblock(int *in, int *out,int *temp, const int blockSize)
{

	int thid = threadIdx.x;
	int gthid = blockIdx.x * blockDim.x + threadIdx.x;
	int blockLen = blockSize;
	int pin = 0, pout = 1;
	temp[pin * blockLen + thid] = in[gthid];
	//temp[pout*len + thid] = (thid > 0) ? temp[pin * len + thid -1]: 0;
	temp[pout*blockLen + thid] = temp[pin*blockLen + thid];
	__syncthreads();
	for (size_t offset = 1; offset < blockLen; offset *= 2)
	{
		if (thid >= offset)
			temp[pout * blockLen + thid] += temp[pout * blockLen + thid - offset];
		__syncthreads();
	}
	if (thid < blockLen)
	{
		out[gthid] = temp[pout * blockLen + thid];
	}
	__syncthreads();
}

__global__ void scanKernel_multiblock(int *in, int *out, int *aux, const int blockSize)
{
	extern __shared__ int temp[];
	int gridID = blockIdx.x;
	int gthid = blockIdx.x * blockDim.x + threadIdx.x;
	scanKernel_singleblock(in, out, temp, blockSize);
	aux[gridID] = out[(gridID) * blockDim.x + blockSize - 1];
	/*
	__syncthreads();
	scanKernel_singleblock(aux, aux,temp, blockSize);
	__syncthreads();
	if (gridID > 0)
	{
		out[gthid] += aux[gridID-1];
	}
	*/
}
__global__ void prefix_add(int *aux, int *input)
{
	int gthid = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x > 0)
	{
		input[gthid] += aux[blockIdx.x - 1];
	}
	
}

int main()
{
	const int len = LENGTH;
	//int in[] = { 0, 0, 3, 4, 0, 6, 6, 7, 0, 7,1,3,3,2,0,1 };
	int *out = new int[len];
	int *scatterResult = new int[len];
	int *in = (int*) malloc(len*sizeof(int));
	for (size_t i = 0; i < len; i++)
	{
		in[i] = i+1;
	}
	//**************warm up**************
	clock_t start, finish;
	double duration;
	int i = TEST_TIMES;
	clearArray(out, len);
	//printArray(in, len);

	//naive method
	start = clock();
	while (i--)
	{
		scanNaive(in, out, len);
	}
	finish = clock();
	duration = finish - start;
	printf("warming up in %f ms\n", duration);
	//*******************timing**********************
	i = TEST_TIMES;
	clearArray(out, len);
	//printArray(in, len);

	//naive method
	start = clock();
	while (i--)
	{
		scanNaive(in, out, len);
	}
	finish = clock();
	duration = finish - start;
	duration = duration / TEST_TIMES;
	//printArray(in, len);
	//printArray(out, len);
	printf("Naive method finished in %f ms\n", duration);
	//shared memory
	i = TEST_TIMES;
	clearArray(out, len);
	start = clock();
	while (i--)
	{
		scanSharedMem_singleblock(in, out, len);
	}
	finish = clock();
	duration = finish - start;
	duration = duration / TEST_TIMES;
	//printArray(in, len);
	//printArray(out, len);
	printf("Shared memory finished in %f ms\n", duration);
	//generalizing
	i = TEST_TIMES;
	clearArray(out, len);
	start = clock();
	while (i--)
	{
		scanSharedMem_multiblock(in, out, len);
	}
	finish = clock();
	duration = finish - start;
	duration = duration / TEST_TIMES;
	//printArray(in, len);
	//printArray(out, len);
	printf("Gerneralized method finished in %f ms\n", duration);
	
	//*****************
	//scattering
	i = TEST_TIMES;
	clearArray(out, len);
	start = clock();
	while (i--)
	{
		scatter(in, scatterResult, len);
	}
	finish = clock();
	duration = finish - start;
	duration = duration / TEST_TIMES;
	//printArray(in, len);
	//printArray(scatterResult, len);
	printf("Scattering finished in %f ms\n", duration);

	/////////thrust
	i = TEST_TIMES;
	clearArray(scatterResult, len);
	start = clock();
	while (i--)
	{
		scatter_thrust(in, scatterResult, len);
	}
	finish = clock();
	duration = finish - start;
	duration = duration / TEST_TIMES;
	printf("thrust scattering finished in %f ms\n", duration);
	//***********************************************

	/*clearArray(out, len);
	printArray(in, len);

	clearArray(out, len);
	scanNaive(in, out, len);
	printArray(out, len);

	clearArray(out, len);
	scanSharedMem_singleblock(in, out, len);
	printArray(out, len);

	clearArray(out, len);
	scanSharedMem_multiblock(in, out, len);
	printArray(out, len);

	scatter(in, scatterResult, len);
	printArray(scatterResult, len);*/
	

	delete[] scatterResult;
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

	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));

	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	
	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);
	
	scanKernel << <1, len >> >(dev_in, dev_out, len);
	
	cudaStatus = cudaMemcpy(out, dev_out, len * sizeof(int), cudaMemcpyDeviceToHost);
	
}

void scanSharedMem_singleblock(const int *in, int *out, int len)
{
	int *dev_in = 0;
	int *dev_out = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	
	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));
	
	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	
	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);
	
	scanKernel_shared_singleblock << <1, len, len * 2 * sizeof(int) >> >(dev_in, dev_out, len);

	cudaStatus = cudaMemcpy(out, dev_out, len * sizeof(int), cudaMemcpyDeviceToHost);

}
void scan_recursive(int *dev_in, int *dev_out, int len)
{
	int blockSize = BLOCK_SIZE;
	int gridSize = ceil(len / (float)blockSize);
	int *current_out = 0;
	int *current_aux = 0;
	cudaMalloc((void**)&current_aux, gridSize * sizeof(int));
	cudaMalloc((void**)&current_out, gridSize * sizeof(int));
	
	if (gridSize <= 1)
	{
		scanKernel_multiblock << <gridSize, blockSize, 2 * blockSize * sizeof(int) >> >(dev_in, dev_out, current_aux, blockSize);
	}
	else
	{
		scanKernel_multiblock << <gridSize, blockSize, 2 * blockSize * sizeof(int) >> >(dev_in, dev_out, current_aux, blockSize);
		scan_recursive(current_aux, current_aux, gridSize);
		prefix_add << <gridSize, blockSize >> >(current_aux, dev_out);
	}
	cudaFree(current_aux);
	cudaFree(current_out);
}
void scanSharedMem_multiblock(const int *in, int *out, int len)
{
	int *dev_in = 0;
	int *dev_out = 0;
	int *dev_aux = 0;

	//define block size and grid size
	int blockSize = BLOCK_SIZE;
	int gridSize = ceil(len / (float)blockSize);
	int length = (len < blockSize) ? len : blockSize;

	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_aux, gridSize* sizeof(int));
	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);

	scan_recursive(dev_in, dev_out, len);
	cudaStatus = cudaMemcpy(out, dev_out, len * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_aux); 
}

__global__ void toFlagKernel(int *in, int *out, int len)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < len)
	{
		if (in[thid] > 0)
			out[thid] = 1;
		else
			out[thid] = 0;
	}
}
__global__ void getScatter(int *in,int *flag, int *out, int len, int *actualLen)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	*actualLen = flag[len - 1];
	out[thid] = 0;
	if (thid < len)
	{
		if (thid == 0)
		{
			if (in[0] > 0)
				out[0] = in[0];
		}
		else if (thid > 0)
		{
			if (flag[thid] > flag[thid - 1])
			{
				out[flag[thid] - 1] = in[thid];
			}
		}
	}
}

void scatter(int *in, int *out, int len)
{
	int *dev_in = 0;
	int *dev_out = 0;
	int *dev_flag = 0;
	int *dev_result = 0;
	int *actualLen = 0;
	
	cudaMalloc((void**)&actualLen, sizeof(int));
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_flag, len* sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_result, len * sizeof(int));

	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);
	
	int blockSize = BLOCK_SIZE;
	int gridSize = ceil(len / (float)blockSize);
	toFlagKernel << <gridSize, blockSize >> >(dev_in, dev_flag, len);
	scan_recursive(dev_flag, dev_out, len);

	getScatter << <gridSize, blockSize >> >(dev_in, dev_out, dev_result, len, actualLen);
	
	cudaStatus = cudaMemcpy(out, dev_result, len * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_flag);
	cudaFree(dev_result);
	cudaFree(actualLen);
}

void scatter_thrust(int *in, int *out, int len)
{
	int *dev_in = 0;
	int *dev_out = 0;
	int *dev_flag = 0;
	int *dev_result = 0;
	int *actualLen = 0;

	cudaMalloc((void**)&actualLen, sizeof(int));
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&dev_in, len * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_out, len* sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_flag, len* sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_result, len * sizeof(int));

	cudaStatus = cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_out, out, len * sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = BLOCK_SIZE;
	int gridSize = ceil(len / (float)blockSize);
	toFlagKernel << <gridSize, blockSize >> >(dev_in, dev_flag, len);

	int *hos_flag = new int[len];
	cudaMemcpy(hos_flag, dev_flag, len * sizeof(int), cudaMemcpyDeviceToHost);

	//scan_recursive(dev_flag, dev_out, len);
	thrust::inclusive_scan(hos_flag,hos_flag+len,hos_flag);
	cudaMemcpy(dev_flag, hos_flag, len * sizeof(int), cudaMemcpyHostToDevice);

	getScatter << <gridSize, blockSize >> >(dev_in, dev_flag, dev_result, len, actualLen);

	cudaStatus = cudaMemcpy(out, dev_result, len * sizeof(int), cudaMemcpyDeviceToHost);

	delete[] hos_flag;
	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_flag);
	cudaFree(dev_result);
	cudaFree(actualLen);
}