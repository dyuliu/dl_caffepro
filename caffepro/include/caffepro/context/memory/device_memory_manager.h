
#pragma once

#include <caffepro/context/memory/memory_manager.h>

namespace caffepro {
	class device_memory_manager : public memory_manager{
	public:
		device_memory_manager(int device_id);
		~device_memory_manager();

	protected:
		// override
		virtual void *sys_alloc(size_t size);
		virtual void sys_free(void *ptr);

	private:
		int device_id_;
	};
}