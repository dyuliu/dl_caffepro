
#include <caffepro/layers/diag_operation_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	diag_operation_layer::diag_operation_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_ALLOW_INPLACE
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	diag_operation_layer::~diag_operation_layer() {
		release_all();
	}

	void diag_operation_layer::init() {
		check_input();

		scale_ = layer_param_.diag_operation_param().scale();
		shift_ = layer_param_.diag_operation_param().shift();
	}

	__global__ static void diag_forward(const int count, const int inner_count, const data_type scale, const data_type shift,
		const data_type *in, data_type *out) {
		CUDA_KERNEL_LOOP(index, count) {
			int i = index % inner_count, n = index / inner_count;
			if (i == n) {
				out[index] = in[index] * scale + shift;
			}
			else {
				out[index] = in[index];
			}
		}
	}

	void diag_operation_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		int count = input.count();
		KERNEL_CALL(diag_forward, count)(count, input.inner_count(), scale_, shift_, input.gpu_data(), outputs_[0]->get(device_index)->mutable_gpu_data());
	}

	__global__ static void diag_backward(const int count, const int inner_count, const data_type scale,
		const data_type *top_diff, data_type *bottom_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, count) {
			int i = index % inner_count, n = index / inner_count;
			data_type v = top_diff[index];
			if (i == n) {
				v *= scale;
			}

			if (scale_targets == 0) {
				bottom_diff[index] = v;
			}
			else {
				bottom_diff[index] = bottom_diff[index] * scale_targets + v;
			}
		}
	}

	void diag_operation_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			data_type beta = get_beta(clear_acts_diff, 0);
			auto &input = *inputs_[0]->get(device_index);
			int count = input.count();

			KERNEL_CALL(diag_backward, count)(count, input.inner_count(), scale_, outputs_[0]->get(device_index)->gpu_diff(), input.mutable_gpu_diff(), beta);
		}
	}
}