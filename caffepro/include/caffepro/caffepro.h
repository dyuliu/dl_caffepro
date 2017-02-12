
#pragma once

#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cudnn/cudnn.h>
#include <curand.h>

// Common definations
typedef float data_type;


// CUDA related 
#define CUDA_NUM_THREADS 1024
#define CUDA_NUM_MAXBLOCKS 65535

#define CUDA_CHECK(condition) CHECK_EQ((condition), cudaSuccess)
#define ENTER_DEVICE_CONTEXT(device_id) { CHECK_GE(device_id, 0); int cur_device_id; cudaGetDevice(&cur_device_id); cudaSetDevice(device_id);
#define EXIT_DEVICE_CONTEXT cudaSetDevice(cur_device_id); }
//#define CUDA_GET_BLOCKS(n) (((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS > CUDA_NUM_MAXBLOCKS ? CUDA_NUM_MAXBLOCKS : ((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
#define CUDA_GET_BLOCKS(n) (((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
#define KERNEL_CALL(kernel_name, n) kernel_name <<< CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS >>>
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK \
if (cudaSuccess != cudaPeekAtLastError()) \
	LOG(FATAL) << "Cuda kernel failed. Error: " \
	<< cudaGetErrorString(cudaPeekAtLastError())

// CUBLAS related
#define CUBLAS_CHECK(condition) CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)

// CURAND related
#define CURAND_CHECK(condition) CHECK_EQ((condition), CURAND_STATUS_SUCCESS)

// CUDNN related
#define CUDNN_CHECK(condition) CHECK_EQ((condition), CUDNN_STATUS_SUCCESS)

// utils

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
	classname(const classname&); \
	classname& operator=(const classname&)

// not implement mark
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"