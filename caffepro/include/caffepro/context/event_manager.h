
#pragma once

#include <pthread/pthread.h>
#include <string>
#include <map>

namespace caffepro {
	class event_manager {
	public:
		// definations
		enum event_type : unsigned int {
			EVENT_TYPE_INIT_DATABASE		= 1U << 0,
			EVENT_TYPE_PREPARE_BATCH		= 1U << 1
		};

		struct event {
			std::string event_name;
			event_type type;
			void *(PTW32_CDECL *start) (void *);
			void *arg;
			pthread_t handle;
		};

	public:
		event_manager();
		~event_manager();

	public:
		// interfaces
		pthread_t create(
			const std::string &name,
			event_type type,
			void *(PTW32_CDECL *start) (void *),
			void *arg
			);

		void wait(const std::string &name);
		void wait(unsigned int event_type_mask);
		void wait_all();

	private:
		std::map<std::string, event> events_;
	};
}