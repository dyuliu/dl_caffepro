
#include <caffepro/context/memory/memory_manager.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	memory_manager::memory_manager() {
		// nothing to do
	}

	memory_manager::~memory_manager() {
		// nothing to do
	}

	void memory_manager::release_all() {
		for (auto pMem = used_.begin(); pMem != used_.end(); ++pMem) {
			sys_free(pMem->second.data);
		}
		used_.clear();

		for (auto pMem = free_.begin(); pMem != free_.end(); ++pMem) {
			sys_free(pMem->second.data);
		}
		free_.clear();
	}

	void *memory_manager::allocate(size_t size) {
		auto pMem = free_.lower_bound(size);
		void *result = nullptr;

		if (pMem != free_.end() && size * 5 > pMem->first) {
			result = pMem->second.data;

			CHECK(!used_.count(pMem->second.get_hash()));
			used_[pMem->second.get_hash()] = pMem->second;
			free_.erase(pMem);
		}
		else {
			memory_block mem;
			mem.data = sys_alloc(size);
			CHECK(mem.data);
			
			mem.size = size;

			CHECK(!used_.count(mem.get_hash()));
			used_[mem.get_hash()] = mem;
			result = mem.data;
		}

		return result;
	}

	void memory_manager::free(void *ptr) {
		memory_hash hMem = memory_block::get_hash(ptr);
		auto pMem = used_.find(hMem);

		if (pMem != used_.end()) {
			free_.insert(std::make_pair(pMem->second.size, pMem->second));
			used_.erase(pMem);
		}
		else {
			LOG(FATAL) << "The pointer does not belong to the memory manager";
		}
	}

	void memory_manager::gc() {
		for (auto pFree = free_.begin(); pFree != free_.end(); ++pFree) {
			sys_free(pFree->second.data);
		}
		free_.clear();
	}
}