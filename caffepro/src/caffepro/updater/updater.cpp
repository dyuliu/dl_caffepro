
#include <caffepro/updater/updater.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	updater::updater(caffepro_context *context, caffepro_net::weight_info &weight_info)
		: context_(context), weight_info_(weight_info) {
		CHECK(context_);
	}

	updater::~updater() {
		// nothing to do
	}

	void updater::update(data_type global_lr, data_type global_wc, bool ignore_zero_lr) {

		if (!ignore_zero_lr || global_lr != 0) {
			merge_diff(ignore_zero_lr);
			for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
				data_type lr = weight_info_.learning_rate[i] * global_lr;
				data_type wc = weight_info_.weight_decay[i] * global_wc;

				if (!ignore_zero_lr || lr != 0) {
					on_update_first_device(i, lr, wc);
				}
			}
			context_->sync_all_devices();
			broadcast_data(ignore_zero_lr);
		}
	}

	void updater::on_update_first_device(int param_id, data_type lr, data_type wc) {
		auto &weight = *weight_info_.weights[param_id]->get(0);
		
		cublas_wrapper<data_type> cublas(context_, weight.device_id());
		cublas.axpby(weight.count(), -lr, weight.gpu_diff(), 1 - lr * wc, weight.mutable_gpu_data());
	}

	void updater::merge_diff(bool ignore_zero_lr) {
		int max_devices = 0;
		for (int param_id = 0; param_id < (int)weight_info_.weights.size(); param_id++) {
			if (!ignore_zero_lr || weight_info_.learning_rate[param_id] != 0) {
				max_devices = std::max(max_devices, (int)weight_info_.weights[param_id]->size());
			}
		}
		
		// binary merge
		for (int step = 1; step < max_devices; step *= 2) {
			for (int param_id = 0; param_id < (int)weight_info_.weights.size(); param_id++) {
				if (!ignore_zero_lr || weight_info_.learning_rate[param_id] != 0) {
					auto &node = *weight_info_.weights[param_id];
					int n_devices = (int)node.size();

					for (int nd = 0; nd + step < n_devices; nd += step * 2) {
						int dest_device_id = node[nd]->device_id();
						cublas_wrapper<data_type> cublas(context_, dest_device_id);
						data_type *buffer = reinterpret_cast<data_type *>(
							context_->get_device(dest_device_id)->memory()->allocate(node[nd]->count() * sizeof(data_type))
							);
						cublas.copy(node[nd]->count(), node[nd + step]->gpu_diff(), buffer);
						cublas.axpy(node[nd]->count(), (data_type)1.f, buffer, node[nd]->mutable_gpu_diff());
						context_->get_device(dest_device_id)->memory()->free(buffer);
					}
				}
			}
			context_->sync_all_devices();
		}
	}

	void updater::broadcast_data(bool ignore_zero_lr) {
		for (int param_id = 0; param_id < (int)weight_info_.weights.size(); param_id++) {
			if (!ignore_zero_lr || weight_info_.learning_rate[param_id] != 0) {
				weight_info_.weights[param_id]->broadcast_data_via_gpu(0);
			}
		}
	}
}