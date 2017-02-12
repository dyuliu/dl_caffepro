
#include <caffepro/layers/relu_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	relu_layer::relu_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_WHEN_INPLACE
			);
	}

	relu_layer::~relu_layer() {
		release_all();
	}

	__global__ void relu_forward(const int n, const data_type* in, data_type* out, float relu_leak) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] > 0 ? in[index] : in[index] * relu_leak;
		}
	}

	void relu_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();
		
		KERNEL_CALL(relu_forward, count)(count, bottom_data, top_data, layer_param_.relu_param().relu_leak());
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ void relu_backward(const int n, const data_type* in_diff,
		const data_type* in_data, data_type* out_diff, float relu_leak, const data_type scale_target) {
		CUDA_KERNEL_LOOP(index, n) {
			data_type v = in_diff[index] * (in_data[index] > 0 ? 1 : relu_leak);
			if (scale_target == 0) {
				out_diff[index] = v;
			}
			else {
				out_diff[index] = out_diff[index] * scale_target + v;
			}
		}
	}

	void relu_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(relu_backward, count)(count, top_diff, bottom_data, bottom_diff, layer_param_.relu_param().relu_leak(), beta_acts);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}