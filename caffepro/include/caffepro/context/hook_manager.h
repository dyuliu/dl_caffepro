
#pragma once 

#include <caffepro/hooks/hook_triggers.h>
#include <vector>
#include <map>

namespace caffepro {

	class caffepro_object;
	class caffepro_context;

	typedef int hook_status;
	typedef unsigned int hook_id;

	class hook_handler_args {
	public:
		hook_handler_args() : content(nullptr) {}

		caffepro_object *content;
	};

	class hook_handler {
	public:
		hook_handler(caffepro_context *context) : context_(context) {}
		virtual ~hook_handler() {}

	public:
		caffepro_context *context() const { return context_; }

	public:
		virtual hook_status notify(hook_trigger trigger, caffepro_object &sender, hook_handler_args &args) = 0;

	protected:
		caffepro_context *context_;
	};

	class hook_manager {
	public:
		hook_manager(caffepro_context *context);
		~hook_manager();

	public:
		hook_id install_hook(hook_trigger trigger, hook_handler *handler);
		bool uninstall_hook(hook_id id);
		std::vector<hook_status> invoke(hook_trigger trigger, caffepro_object &sender, hook_handler_args &args);

	protected:
		hook_id alloc_hook_id();

	protected:
		caffepro_context *context_;
		hook_id next_hook_id_;
		std::map<hook_trigger, std::map<hook_id, hook_handler *> > hooks_;
	};
}