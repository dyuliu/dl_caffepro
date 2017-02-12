
#include <caffepro/layers/sigmoid_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	sigmoid_layer::sigmoid_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			);
	}

	sigmoid_layer::~sigmoid_layer() {
		release_all();
	}

	__device__ inline data_type sigmoid_gpu(data_type x) {
		return 1. / (1. + exp(-x));
	}

	__global__ static void sigmoid_forward(const int n, const data_type* in, data_type* out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = sigmoid_gpu(in[index]);
		}
	}

	void sigmoid_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();

		KERNEL_CALL(sigmoid_forward, count)(count, bottom_data, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void sigmoid_backward(const int n, const data_type* in_diff, const data_type* in_data, data_type* out_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, n) {
			data_type sigmoid_x = sigmoid_gpu(in_data[index]);
			data_type v = in_diff[index] * sigmoid_x * (1 - sigmoid_x);

			if (scale_targets == 0) {
				out_diff[index] = v;
			}
			else {
				out_diff[index] = scale_targets * out_diff[index] + v;
			}
		}
	}

	void sigmoid_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(sigmoid_backward, count)(count, top_diff, bottom_data, bottom_diff, beta_acts);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}