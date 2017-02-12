
#pragma once

#include <map>

namespace caffepro {

	class memory_manager {
	protected:
		// typedef
		typedef unsigned long long memory_hash;

		struct memory_block {
			void *data;
			size_t size;
			unsigned int flag;

			memory_block() : data(nullptr), size(0), flag(0) {}
			memory_hash get_hash() { return get_hash(data); }
			static memory_hash get_hash(void *ptr) { return reinterpret_cast<memory_hash>(ptr); }
		};

	public:
		memory_manager();
		virtual ~memory_manager();

	protected:
		// call this when the derived class destructs
		void release_all();

	public:
		// methods
		void *allocate(size_t size);
		void free(void *ptr);
		void gc();

	protected:
		// override
		virtual void *sys_alloc(size_t size) = 0;
		virtual void sys_free(void *ptr) = 0;

	protected:
		// member
		std::multimap<size_t, memory_block> free_;
		std::map<memory_hash, memory_block> used_;
	};

}