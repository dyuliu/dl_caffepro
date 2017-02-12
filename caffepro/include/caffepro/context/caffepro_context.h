
#pragma once

#include <caffepro/context/device_context.h>
#include <caffepro/context/event_manager.h>
#include <caffepro/context/hook_manager.h>
#include <caffepro/caffepro.h>
#include <vector>
#include <map>

namespace caffepro {

	class caffepro_object;

	class caffepro_context {
	public:
		// definations
		typedef std::vector<device_context *> device_list;
		typedef std::map<std::string, std::string> str_settings;
		typedef std::map<std::string, caffepro_object *> caffepro_object_dic;
		enum phase_type {TRAIN, TEST};
		enum signal_type : unsigned int {
			SIGNAL_STOP_ALL = 1U << 0
		};

	public:
		caffepro_context();
		~caffepro_context();

	public:
		// interfaces
		device_context* get_device(int device_id);
		device_context* get_current_device();
		void sync_all_devices();
		caffepro_object *get_shared_object(const std::string &ns, const std::string &key) const;
		caffepro_object *set_shared_object(const std::string &ns, const std::string &key, caffepro_object *obj);

	public:
		// fetch functions
		phase_type get_phase() const { return phase_; }
		void set_phase(phase_type type) { phase_ = type; }
		event_manager* events() { return &events_; }
		hook_manager* hooks() { return &hooks_; }
		std::string get_global_cfg(const std::string &key) { return global_settings_[key]; }
		void set_global_cfg(const std::string &key, const std::string &val) { global_settings_[key] = val; }
		unsigned int get_signal(unsigned int mask) const { return signal_ & mask; }
		void set_signal(unsigned int mask) { signal_ |= mask; }
		void clear_signal(unsigned int mask) { signal_ &= ~mask; }

	private:
		device_list devices_;
		phase_type phase_;
		event_manager events_;
		str_settings global_settings_;
		hook_manager hooks_;
		std::map<std::string, caffepro_object_dic> shared_object_;
		unsigned int signal_;

	private:
		DISABLE_COPY_AND_ASSIGN(caffepro_context);
	};

}