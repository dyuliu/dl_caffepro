
#include <caffepro/updater/sgd_updater.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	sgd_updater::sgd_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics)
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

	sgd_updater::~sgd_updater() {
		// nothing to do
	}

	void sgd_updater::load_updater_state(SolverState &state) {
		// load solver state
		CHECK_EQ(state.history_size(), history_.size()) << "Incorrect length of history blobs.";
		LOG(INFO) << "SGDSolver: restoring history";

		for (int i = 0; i < (int)history_.size(); ++i) {
			history_[i]->load_data_from(state.mutable_history(i));
		}
	}

	void sgd_updater::save_updater_state(SolverState &state) {
		state.clear_history();
		for (int i = 0; i < history_.size(); ++i) {
			// Add history
			BlobProto* history_blob = state.add_history();
			history_[i]->save_data_to(history_blob);
		}
	}

	void sgd_updater::on_update_first_device(int param_id, data_type lr, data_type wc) {
		device_blob &weight = *weight_info_.weights[param_id]->get(0);
		device_blob &his = *history_[param_id];
		cublas_wrapper<data_type> cublas(context_, weight.device_id());

		//LOG(ERROR) << weight.mean(false) << " " << weight.variance(false) << " " << weight.mean(true) << " " << weight.variance(true);

		cublas.axpby(weight.count(), lr * wc, weight.gpu_data(), momentum_, his.mutable_gpu_data());
		cublas.axpy(weight.count(), lr, weight.gpu_diff(), his.mutable_gpu_data());
		cublas.axpy(weight.count(), (data_type)-1.f, his.gpu_data(), weight.mutable_gpu_data());
	}
}