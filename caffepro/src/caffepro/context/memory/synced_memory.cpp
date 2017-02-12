
#include <caffepro/context/memory/syncd_memory.h>
#include <memory>

namespace caffepro {
	synced_memory::synced_memory(const size_t size, memory_manager *manager, int device_id)
		: cpu_capacity_(0), gpu_capacity_(0), head_(UNINITIALIZED),
			cpu_data_(nullptr), gpu_data_(nullptr), size_(size), 
			mem_manager_(manager), own_dev_data_(true) {

		if (device_id < 0) {
			CUDA_CHECK(cudaGetDevice(&device_id));
		}
		
		device_id_ = device_id;

		if (manager) {
			usage_ = SYNCMEM_USAGE_MANAGED;
		}
		else {
			usage_ = SYNCMEM_USAGE_UNMANAGED;
		}
	}

	synced_memory::~synced_memory() {
		release_all();
	}

	void synced_memory::to_cpu() {
		switch (head_) {
		case UNINITIALIZED:
			cpu_resize();
			head_ = HEAD_AT_CPU;
			break;

		case HEAD_AT_GPU:
			gpu_resize();
			cpu_resize();
			ENTER_DEVICE_CONTEXT(device_id_);
				CUDA_CHECK(cudaMemcpy(cpu_data_, gpu_data_, size_, cudaMemcpyDeviceToHost));
			EXIT_DEVICE_CONTEXT;
			head_ = SYNCED;
			break;

		case HEAD_AT_CPU:
			cpu_resize();
			break;

		case SYNCED:
			if (cpu_resize()) {
				head_ = HEAD_AT_CPU;
			}
			break;
		}
	}

	void synced_memory::to_gpu() {

		switch (head_) {
		case UNINITIALIZED:
			gpu_resize();
			head_ = HEAD_AT_GPU;
			break;

		case HEAD_AT_CPU:
			cpu_resize();
			gpu_resize();
			ENTER_DEVICE_CONTEXT(device_id_);
				CUDA_CHECK(cudaMemcpy(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice));
			EXIT_DEVICE_CONTEXT;
			head_ = SYNCED;
			break;

		case HEAD_AT_GPU:
			gpu_resize();
			break;

		case SYNCED:
			if (gpu_resize()) {
				head_ = HEAD_AT_GPU;
			}
			break;
		}
	}

	void synced_memory::to_gpu_async() {

		switch (head_) {
		case UNINITIALIZED:
			gpu_resize();
			head_ = HEAD_AT_GPU;
			break;

		case HEAD_AT_CPU:
			cpu_resize();
			gpu_resize();
			ENTER_DEVICE_CONTEXT(device_id_);
				CUDA_CHECK(cudaMemcpyAsync(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice));
			EXIT_DEVICE_CONTEXT;
			head_ = SYNCED;
			break;

		case HEAD_AT_GPU:
			gpu_resize();
			break;

		case SYNCED:
			if (gpu_resize()) {
				head_ = HEAD_AT_GPU;
			}
			break;
		}
	}

	const void* synced_memory::cpu_data() {
		to_cpu();
		return static_cast<const void*>(cpu_data_);
	}

	const void* synced_memory::gpu_data() {
		to_gpu();
		return static_cast<const void*>(gpu_data_);
	}

	const void* synced_memory::async_gpu_data() {
		to_gpu_async();
		return static_cast<const void*>(gpu_data_);
	}

	void* synced_memory::mutable_cpu_data() {
		to_cpu();
		head_ = HEAD_AT_CPU;
		return cpu_data_;
	}

	void* synced_memory::mutable_gpu_data() {
		to_gpu();
		head_ = HEAD_AT_GPU;
		return gpu_data_;
	}

	void* synced_memory::write_only_cpu_data() {
		cpu_resize();
		head_ = HEAD_AT_CPU;
		return cpu_data_;
	}

	void* synced_memory::write_only_gpu_data() {
		gpu_resize();
		head_ = HEAD_AT_GPU;
		return gpu_data_;
	}

	void synced_memory::release_all() {
		if (cpu_data_) {
			free(cpu_data_);
			cpu_data_ = nullptr;
		}

		if (gpu_data_ && own_dev_data_) {
			ENTER_DEVICE_CONTEXT(device_id_)
				if (usage_ == SYNCMEM_USAGE_MANAGED) {
					CHECK(mem_manager_);
					mem_manager_->free(gpu_data_);
				}
				else if (usage_ == SYNCMEM_USAGE_UNMANAGED) {
					CUDA_CHECK(cudaFree(gpu_data_));
				}
				gpu_data_ = nullptr;
			EXIT_DEVICE_CONTEXT;
		}

		head_ = UNINITIALIZED;
		cpu_capacity_ = 0;
		gpu_capacity_ = 0;
		own_dev_data_ = true;
	}

	void synced_memory::copy_from_via_cpu(synced_memory &other) {
		CHECK_EQ(size_, other.size_);
		
		memcpy(mutable_cpu_data(), other.cpu_data(), size_);
	}

	void synced_memory::copy_from_via_gpu(synced_memory &other) {
		CHECK_EQ(size_, other.size_);
	
		ENTER_DEVICE_CONTEXT(device_id_)
			CUDA_CHECK(cudaMemcpyAsync(mutable_gpu_data(), other.gpu_data(), size_, cudaMemcpyDeviceToDevice));
		EXIT_DEVICE_CONTEXT;
	}

	void synced_memory::use_external_dev_memory(void *dev_mem, size_t capacity) {
		CHECK_GE(capacity, size_);
		mutable_cpu_data(); // head to cpu
		
		if (gpu_data_ && own_dev_data_) { // clear original data
			ENTER_DEVICE_CONTEXT(device_id_)
				if (usage_ == SYNCMEM_USAGE_MANAGED) {
					CHECK(mem_manager_);
					mem_manager_->free(gpu_data_);
				}
				else if (usage_ == SYNCMEM_USAGE_UNMANAGED) {
					CUDA_CHECK(cudaFree(gpu_data_));
				}
			EXIT_DEVICE_CONTEXT;
		}

		gpu_data_ = dev_mem;
		gpu_capacity_ = capacity;
		own_dev_data_ = false;
	}

	void synced_memory::use_internal_dev_memory() {
		if (!own_dev_data_) {
			mutable_cpu_data(); // head to cpu
			gpu_data_ = nullptr;
			gpu_capacity_ = 0;
			own_dev_data_ = true;
		}
	}

	size_t synced_memory::cpu_resize() {
		if (!cpu_data_) {
			cpu_data_ = malloc(size_);
		}
		else if (size_ > cpu_capacity_) {
			cpu_data_ = realloc(cpu_data_, size_);
		}
		else {
			return 0;
		}

		size_t num_new_bytes = size_ - cpu_capacity_;
		cpu_capacity_ = size_;

		return num_new_bytes;
	}

	size_t synced_memory::gpu_resize() {

		size_t num_new_bytes = 0;

		ENTER_DEVICE_CONTEXT(device_id_)

			if (!gpu_data_) {
				CHECK(own_dev_data_);
				if (usage_ == SYNCMEM_USAGE_MANAGED) {
					CHECK(mem_manager_);
					gpu_data_ = mem_manager_->allocate(size_);
				}
				else if (usage_ == SYNCMEM_USAGE_UNMANAGED) {
					CUDA_CHECK(cudaMalloc(&gpu_data_, size_)) << "on device " << device_id_;
				}
				num_new_bytes = size_ - gpu_capacity_;
				gpu_capacity_ = size_;
			}
			else if (size_ > gpu_capacity_) {
				CHECK(own_dev_data_);
				void* new_gpu_data = nullptr;
				if (usage_ == SYNCMEM_USAGE_MANAGED) {
					CHECK(mem_manager_);
					new_gpu_data = mem_manager_->allocate(size_);
					mem_manager_->free(gpu_data_);
				}
				else if (usage_ == SYNCMEM_USAGE_UNMANAGED) {
					CUDA_CHECK(cudaMalloc(&new_gpu_data, size_)) << "on device " << device_id_;
					CUDA_CHECK(cudaFree(gpu_data_));
				}

				gpu_data_ = new_gpu_data;
				num_new_bytes = size_ - gpu_capacity_;
				gpu_capacity_ = size_;
			}
			else {

				//if (size_ < gpu_capacity_ && usage_ == SYNCMEM_USAGE_MANAGED) {
				//	CHECK(mem_manager_);
				//	mem_manager_->free(gpu_data_);
				//	gpu_data_ = mem_manager_->allocate(size_);
				//	gpu_capacity_ = size_;
				//}

				num_new_bytes = 0;
			}

		EXIT_DEVICE_CONTEXT;

		return num_new_bytes;
	}
}