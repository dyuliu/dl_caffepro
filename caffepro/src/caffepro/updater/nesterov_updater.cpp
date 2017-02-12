
#include <caffepro/updater/nesterov_updater.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	nesterov_updater::nesterov_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics)
		: sgd_updater_faster(context, weight_info, param, update_metrics) {
	}

	nesterov_updater::~nesterov_updater() {
		// nothing to do
	}

	void nesterov_updater::on_update_first_device(int param_id, data_type lr, data_type wc) {
		device_blob &weight = *update_groups_[param_id].group_data->get(0);
		device_blob &his = *update_groups_[param_id].group_history;
		cublas_wrapper<data_type> cublas(context_, weight.device_id());

		cublas.axpy(weight.count(), wc, weight.gpu_data(), weight.mutable_gpu_diff());
		cublas.axpby(weight.count(), 1.f, weight.gpu_diff(), momentum_, his.mutable_gpu_data());
		cublas.axpy(weight.count(), momentum_, his.gpu_data(), weight.mutable_gpu_diff());

		cublas.axpy(weight.count(), -lr, weight.gpu_diff(), weight.mutable_gpu_data());
	}
}