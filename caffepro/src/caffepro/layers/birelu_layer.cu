
#include <caffepro/layers/birelu_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	birelu_layer::birelu_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_WHEN_INPLACE
			);
	}

	birelu_layer::~birelu_layer() {
		release_all();
	}

	void birelu_layer::resize() {
		check_input();

		bool init = (outputs_[0]->size() == 0);
		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);

			if (input.reshaped()) {
				if (init) {
					outputs_[0]->set_4d(nd,
						input.num(),
						input.channels() * 2,
						input.height(),
						input.width(),
						input.device_id(),
						context_);
				}
				else {
					NOT_IMPLEMENTED;
				}
			}
		}
	}

	__global__ void birelu_forward(const int n, const data_type* in, data_type* out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] > 0 ? in[index] : 0;
			out[n + index] = in[index] <= 0 ? in[index] : 0;
		}
	}

	void birelu_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();
		
		KERNEL_CALL(birelu_forward, count)(count, bottom_data, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ void birelu_backward(const int n, const data_type* top_diff,
		const data_type* bottom_data, data_type* bottom_diff, const data_type scale_target) {
		CUDA_KERNEL_LOOP(index, n) {
			data_type v = (bottom_data[index] > 0) ?
				top_diff[index] : top_diff[n + index];
			if (scale_target == 0) {
				bottom_diff[index] = v;
			}
			else {
				bottom_diff[index] = bottom_diff[index] * scale_target + v;
			}
		}
	}

	void birelu_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(birelu_backward, count)(count, top_diff, bottom_data, bottom_diff, beta_acts);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}