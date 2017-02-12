
#include <caffepro/context/hook_manager.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	hook_manager::hook_manager(caffepro_context *context)
		: context_(context), next_hook_id_(hook_id()) {
		// nothing to do
	}

	hook_manager::~hook_manager() {
		// nothing to do
	}

	hook_id hook_manager::install_hook(hook_trigger trigger, hook_handler *handler) {
		CHECK(handler);
		CHECK_EQ(handler->context(), context_) << "Unable to install hook because they belong to differnet contexts";

		hook_id id = alloc_hook_id();
		hooks_[trigger][id] = handler;

		return id;
	}

	bool hook_manager::uninstall_hook(hook_id id) {
		for (auto &hk_group : hooks_) {
			if (hk_group.second.count(id)) {
				hk_group.second.erase(id);
				return true; // hook id is unique
			}
		}
		return false;
	}

	std::vector<hook_status> hook_manager::invoke(hook_trigger trigger, caffepro_object &sender, hook_handler_args &args) {
		std::vector<hook_status> status;

		for (auto &hk : hooks_[trigger]) {
			status.push_back(hk.second->notify(trigger, sender, args));
		}

		return status;
	}

	hook_id hook_manager::alloc_hook_id() {
		return next_hook_id_++;
	}
}