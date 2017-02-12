
#include <caffepro/context/event_manager.h>
#include <caffepro/caffepro.h>
#include <vector>

namespace caffepro {
	event_manager::event_manager() {
		// nothing to do
	}

	event_manager::~event_manager() {
		wait_all();
	}

	pthread_t event_manager::create(const std::string &name, event_type type,
		void *(PTW32_CDECL *start) (void *), void *arg) {
		event evt;
		evt.event_name = name;
		evt.type = type;
		evt.start = start;
		evt.arg = arg;

		if (events_.count(name)) {
			// wait for another event with the same name finishs
			wait(name);
		}

		CHECK(!pthread_create(&evt.handle, nullptr, start, arg))
			<< "Pthread execution failed.";
		events_[name] = evt;
		return evt.handle;
	}

	void event_manager::wait(const std::string &name) {
		if (events_.count(name)) {
			pthread_join(events_[name].handle, nullptr);
			events_.erase(name);
		}
	}

	void event_manager::wait(unsigned int event_type_mask) {
		std::vector<std::string> events_to_remove;
		for (auto iter = events_.begin(); iter != events_.end(); ++iter) {
			if (iter->second.type | event_type_mask) {
				pthread_join(iter->second.handle, nullptr);
				events_to_remove.push_back(iter->first);
			}
		}

		for (auto iter = events_to_remove.begin(); iter != events_to_remove.end(); ++iter) {
			events_.erase(*iter);
		}
	}

	void event_manager::wait_all() {
		for (auto iter = events_.begin(); iter != events_.end(); ++iter) {
			pthread_join(iter->second.handle, nullptr);
		}
		events_.clear();
	}
}