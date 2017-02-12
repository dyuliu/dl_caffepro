
#include <caffepro/context/device_context.h>
#include <caffepro/caffepro.h>

#include <process.h>
#include <cstdio>

#include <caffepro/context/memory/device_memory_manager.h>

namespace caffepro {

	inline long long cluster_seedgen(void) {
		long long s, seed, pid;
		pid = _getpid();
		s = time(NULL);
		seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
		return seed;
	}

	device_context::device_context(int device_id) 
		: device_id_(device_id) {

		ENTER_DEVICE_CONTEXT(device_id)
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));

			CURAND_CHECK(curandCreateGenerator(&curand_handle_, CURAND_RNG_PSEUDO_DEFAULT));
			CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_handle_, cluster_seedgen()));

			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));

			memory_ = new device_memory_manager(device_id);

		EXIT_DEVICE_CONTEXT;
	}

	device_context::~device_context() {
		CUBLAS_CHECK(cublasDestroy(cublas_handle_));
		CURAND_CHECK(curandDestroyGenerator(curand_handle_));
		CUDNN_CHECK(cudnnDestroy(cudnn_handle_));

		delete memory_;
		memory_ = nullptr;
	}

	device_context *device_context::create(int device_id) {
		return new device_context(device_id);
	}
}