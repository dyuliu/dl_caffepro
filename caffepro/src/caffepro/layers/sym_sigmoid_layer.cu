
#include <caffepro/layers/sym_sigmoid_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	sym_sigmoid_layer::sym_sigmoid_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS
			);
	}

	sym_sigmoid_layer::~sym_sigmoid_layer() {
		release_all();
	}

	__device__ inline data_type sym_sigmoid_gpu(data_type x) {
		return 2. / (1. + exp(-x)) - 1;
	}

	__global__ static void sym_sigmoid_forward(const int n, const data_type* in, data_type* out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = sym_sigmoid_gpu(in[index]);
		}
	}

	void sym_sigmoid_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();

		KERNEL_CALL(sym_sigmoid_forward, count)(count, bottom_data, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void sym_sigmoid_backward(const int n, const data_type* in_diff, const data_type* out_data, data_type* out_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, n) {
			data_type y = out_data[index];
			data_type v = 0.5f * in_diff[index] * (1 - y * y);

			if (scale_targets == 0) {
				out_diff[index] = v;
			}
			else {
				out_diff[index] = scale_targets * out_diff[index] + v;
			}
		}
	}

	void sym_sigmoid_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* top_data = outputs_[0]->get(device_index)->gpu_data();
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(sym_sigmoid_backward, count)(count, top_diff, top_data, bottom_diff, beta_acts);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}