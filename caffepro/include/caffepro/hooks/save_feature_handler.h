
#pragma once

#include <caffepro/context/hook_manager.h>
#include <string>
#include <fstream>

namespace caffepro {
	class save_feature_handler : public hook_handler {
	public:
		save_feature_handler(caffepro_context *context);
		virtual ~save_feature_handler();

	public:
		void open(const std::string &layer_name, const std::string &file_name);
		void close();
		virtual hook_status notify(hook_trigger trigger, caffepro_object &sender, hook_handler_args &args);
	
	protected:
		std::ofstream stream_;
		std::string layer_name_;
		bool first_run_;
	};
}