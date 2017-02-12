
#include <caffepro/layers/householder_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	householder_layer::householder_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	householder_layer::~householder_layer() {
		release_all();
	}

	void householder_layer::init() {
		check_input();

		source_ = layer_param_.householder_param().source();
	}

	void householder_layer::resize() {
		check_input();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				CHECK_LT(source_, inputs_[0]->get(nd)->num());
				outputs_[0]->set_4d(nd, inputs_[0]->get(nd)->inner_count(), inputs_[0]->get(nd)->channels(),
					inputs_[0]->get(nd)->height(), inputs_[0]->get(nd)->width(), inputs_[0]->get(nd)->device_id(),
					context_);
			}
		}
	}

	__global__ static void householder_fwd_kernel(const int n, const int input_len, const data_type *inputs, data_type *outputs) {
		CUDA_KERNEL_LOOP(index, n) {
			int row = index / input_len, col = index % input_len;
			data_type v = -2 * inputs[row] * inputs[col];
			if (row == col) v = 1 + v;
			outputs[index] = v;
		}
	}

	void householder_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		KERNEL_CALL(householder_fwd_kernel, output.count())(output.count(), input.inner_count(),
			input.gpu_data() + input.offset(source_), output.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
	}

	void householder_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			auto &input = *inputs_[0]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);
			data_type beta = get_beta(clear_acts_diff, 0);
			if (beta == 0) {
				input.fill_diff(0.f); // clear diff
			}

			data_type *target_diff = input.mutable_gpu_diff() + input.offset(source_);
			const data_type *input_data = input.gpu_data() + input.offset(source_);
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.gemv(CblasNoTrans, output.num(), output.inner_count(), -2.f, output.gpu_diff(), input_data, beta, target_diff);
			cublas.gemv(CblasTrans, output.num(), output.inner_count(), -2.f, output.gpu_diff(), input_data, 1.f, target_diff);
		}
	}
}