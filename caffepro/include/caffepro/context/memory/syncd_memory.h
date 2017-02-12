
#pragma once

#include <caffepro/context/memory/memory_manager.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	class synced_memory {
	public:
		// definations
		enum synced_memory_usage {
			SYNCMEM_USAGE_MANAGED,
			SYNCMEM_USAGE_UNMANAGED
		};

		enum synced_head {
			UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
		};

	public:
		synced_memory(const size_t size, memory_manager *manager = nullptr, int device_id = -1);
		~synced_memory();

	public:
		// interfaces
		const void* cpu_data();
		const void* gpu_data();
		const void* async_gpu_data();
		void* mutable_cpu_data();
		void* mutable_gpu_data();
		void* write_only_cpu_data();
		void* write_only_gpu_data();
		void release_all();

		void copy_from_via_cpu(synced_memory &other);
		void copy_from_via_gpu(synced_memory &other);

		// original data will be copied to the new memory
		void use_external_dev_memory(void *dev_mem, size_t capacity);
		void use_internal_dev_memory();

	public:
		// fetch functions
		inline size_t size() const { return size_; }
		inline void set_size(const size_t size) { size_ = size; }
		inline size_t cpu_capacity() const { return cpu_capacity_; }
		inline size_t gpu_capacity() const { return gpu_capacity_; }
		inline int device_id() const { return device_id_; }

	private:
		// internal methods
		void to_cpu();
		void to_gpu();
		void to_gpu_async();
		size_t cpu_resize();
		size_t gpu_resize();

	private:
		// members
		void* cpu_data_;
		void* gpu_data_;
		size_t cpu_capacity_;
		size_t gpu_capacity_;
		size_t size_;
		synced_head head_;
		synced_memory_usage usage_;
		int device_id_;
		memory_manager *mem_manager_;
		bool own_dev_data_;

	private:
		DISABLE_COPY_AND_ASSIGN(synced_memory);
	};

}