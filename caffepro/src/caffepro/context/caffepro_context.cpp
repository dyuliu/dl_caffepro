
#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>
#include <random>
#include <omp.h>

namespace caffepro {

	caffepro_context::caffepro_context() 
		: phase_(TEST), signal_(0), hooks_(this) {

		// init random seed for every threads
		srand(std::random_device()());

#pragma omp parallel for
		for (int i = 0; i < 64; i++) {
			int id = omp_get_thread_num();
			int rnd = std::random_device()();
			srand(id + rnd);
		}
	}

	caffepro_context::~caffepro_context() {
		for (auto pDev = devices_.begin(); pDev != devices_.end(); ++pDev) {
			if (*pDev) {
				delete *pDev;
				*pDev = nullptr;
			}
		}
	}

	device_context* caffepro_context::get_device(int device_id) {
		CHECK_GE(device_id, 0);

		if (device_id >= (int)devices_.size()) {
			for (int i = (int)devices_.size(); i <= device_id; i++) {
				devices_.push_back(nullptr);
			}
		}
		if (!devices_[device_id]) {
			devices_[device_id] = device_context::create(device_id);
		}
		return devices_[device_id];
	}

	device_context* caffepro_context::get_current_device() {
		int device_id;
		CUDA_CHECK(cudaGetDevice(&device_id));
		return get_device(device_id);
	}

	void caffepro_context::sync_all_devices() {
		for (int device_id = 0; device_id < (int)devices_.size(); device_id++) {
			if (devices_[device_id]) {
				ENTER_DEVICE_CONTEXT(device_id);
					CUDA_CHECK(cudaDeviceSynchronize());
				EXIT_DEVICE_CONTEXT;
			}
		}
	}

	caffepro_object *caffepro_context::get_shared_object(const std::string &ns, const std::string &key) const {
		if (shared_object_.count(ns) && shared_object_.find(ns)->second.count(key)) {
			return shared_object_.find(ns)->second.find(key)->second;
		}
		
		return nullptr;
	}

	caffepro_object *caffepro_context::set_shared_object(const std::string &ns, const std::string &key, caffepro_object *obj) {
		caffepro_object *old = get_shared_object(ns, key);
		shared_object_[ns][key] = obj;
		return old;
	}
}