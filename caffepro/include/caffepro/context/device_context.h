
#pragma once

#include <cublas_v2.h>
#include <curand.h>
#include <cudnn/cudnn.h>

#include <caffepro/context/memory/memory_manager.h>

namespace caffepro {

	class device_context {
	private:
		device_context(int device_id);

	public:
		~device_context();

	public:
		// factory function
		static device_context *create(int device_id);

	public:
		// fetch functions
		cublasHandle_t cublas_handle() const { return cublas_handle_; }
		cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }
		curandGenerator_t curand_handle() const { return curand_handle_; }
		int device_id() const { return device_id_; }
		memory_manager *memory() { return memory_; }

	private:
		int device_id_;
		cublasHandle_t cublas_handle_;
		cudnnHandle_t cudnn_handle_;
		curandGenerator_t curand_handle_;
		memory_manager *memory_;
	};

}