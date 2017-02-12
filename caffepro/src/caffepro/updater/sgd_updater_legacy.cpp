
#include <caffepro/updater/sgd_updater_legacy.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	sgd_updater_legacy::sgd_updater_legacy(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics)
		: updater(context, weight_info), momentum_(param.momentum()) {

		history_.resize(weight_info_.weights.size());
		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			history_[i].reset(device_blob::create_like(
				context_,
				*weight_info_.weights[i]->get(0),
				weight_info_.weights[i]->get(0)->device_id()
				));
			history_[i]->fill_data((data_type)0.f);
		}
	}

	sgd_updater_legacy::~sgd_updater_legacy() {
		// nothing to do
	}

	void sgd_updater_legacy::load_updater_state(SolverState &state) {
		// load solver state
		CHECK_EQ(state.history_size(), history_.size()) << "Incorrect length of history blobs.";
		LOG(INFO) << "SGDSolver: restoring history";

		for (int i = 0; i < (int)history_.size(); ++i) {
			history_[i]->load_data_from(state.mutable_history(i));
		}
	}

	void sgd_updater_legacy::save_updater_state(SolverState &state) {
		state.clear_history();
		for (int i = 0; i < history_.size(); ++i) {
			// Add history
			BlobProto* history_blob = state.add_history();
			history_[i]->save_data_to(history_blob);
		}
	}

	void sgd_updater_legacy::update(data_type global_lr, data_type global_wc, bool ignore_zero_lr) {
		if (!ignore_zero_lr || global_lr != 0) {
			for (int param_id = 0; param_id < (int)weight_info_.weights.size(); param_id++) {
				if (!ignore_zero_lr || weight_info_.learning_rate[param_id] != 0) {
					node_blob &node = *weight_info_.weights[param_id];
					device_blob &his = *history_[param_id];

					data_type lr = weight_info_.learning_rate[param_id] * global_lr;
					data_type wc = weight_info_.weight_decay[param_id] * global_wc;
					cublas_wrapper<data_type> cublas(context_, node[0]->device_id());

					// first, v(t) <- v(t-1) * mom + lr * wc * w
					if (wc != 0) {
						cublas.axpby(
							node[0]->count(),
							lr * wc,
							node[0]->gpu_data(), 
							momentum_,
							his.mutable_gpu_data()
							);
					}
					else // scale v(t - 1) only
					{
						cublas.scal(
							node[0]->count(), 
							momentum_,
							his.mutable_gpu_data()
							);
					}

					// second, for each gpu, v(t) += lr * gradient
					for (int nd = 0; nd < node.size(); nd++) {
						if (nd > 0) { // copy diff to gpu[0]
							cublas.copy(
								node[nd]->count(),
								node[nd]->gpu_diff(),
								node[0]->mutable_gpu_diff()
								);
						}

						cublas.axpy(
							node[0]->count(), 
							lr,
							node[0]->gpu_diff(),
							his.mutable_gpu_data()
							);
					}

					// at last, update
					cublas.axpy(
						his.count(),
						(data_type)-1.f,
						his.gpu_data(),
						node[0]->mutable_gpu_data()
						);

					node.broadcast_data_via_gpu(0);
				}
			}
		}
	}
}