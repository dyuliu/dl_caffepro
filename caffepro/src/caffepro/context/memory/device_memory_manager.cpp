
#include <caffepro/context/memory/device_memory_manager.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	device_memory_manager::device_memory_manager(int device_id)
		: device_id_(device_id) {
	}

	device_memory_manager::~device_memory_manager() {
		release_all();
	}

	void *device_memory_manager::sys_alloc(size_t size) {
		void *result = nullptr;

		ENTER_DEVICE_CONTEXT(device_id_)
			if (cudaMalloc(&result, size) != cudaSuccess) {
				cudaGetLastError(); // clear error stack
				gc();
				CUDA_CHECK(cudaMalloc(&result, size));
			}
		EXIT_DEVICE_CONTEXT;

		return result;
	}

	void device_memory_manager::sys_free(void *ptr) {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUDA_CHECK(cudaFree(ptr));
		EXIT_DEVICE_CONTEXT;
	}
}