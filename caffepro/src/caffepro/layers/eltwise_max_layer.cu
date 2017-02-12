
#include <caffepro/layers/eltwise_max_layer.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	eltwise_max_layer::eltwise_max_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = 2;
		attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_SAME_COUNT
			| layer_attribute::CF_REQUIRE_SAME_NUM
			);
	}

	eltwise_max_layer::~eltwise_max_layer() {
		release_all();
	}

	void eltwise_max_layer::init() {
		check_input();
	}

	__global__ static void eltmax_forward(int n, const data_type *input1, const data_type *input2, data_type *output) {
		CUDA_KERNEL_LOOP(index, n) {
			output[index] = fmaxf(input1[index], input2[index]);
		}
	}

	void eltwise_max_layer::on_forward(int device_index) {
		auto &input1 = *inputs_[0]->get(device_index);
		auto &input2 = *inputs_[1]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		KERNEL_CALL(eltmax_forward, input1.count())(input1.count(), input1.gpu_data(), input2.gpu_data(), output.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void eltmax_backward(int n, const data_type *input1_data, const data_type *input2_data, const data_type *diff,
		data_type *input1_diff, data_type *input2_diff, const data_type scale_target1, const data_type scale_target2, bool bp1, bool bp2) {
		CUDA_KERNEL_LOOP(index, n) {
			if (input1_data[index] >= input2_data[index]) {
				if (bp1) {
					if (scale_target1 == 0) {
						input1_diff[index] = diff[index];
					}
					else {
						input1_diff[index] = diff[index] + input1_diff[index] * scale_target1;
					}
				}
				if (bp2) {
					if (scale_target2 == 0) {
						input2_diff[index] = 0;
					}
					else {
						input2_diff[index] = input2_diff[index] * scale_target2;
					}
				}
			}
			else {
				if (bp1) {
					if (scale_target1 == 0) {
						input1_diff[index] = 0;
					}
					else {
						input1_diff[index] = input1_diff[index] * scale_target1;
					}
				}
				if (bp2) {
					if (scale_target2 == 0) {
						input2_diff[index] = diff[index];
					}
					else {
						input2_diff[index] = diff[index] + input2_diff[index] * scale_target2;
					}
				}
			}
		}
	}

	void eltwise_max_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0) || should_bp(bp_acts, 1)) {
			bool bp1 = should_bp(bp_acts, 0);
			bool bp2 = should_bp(bp_acts, 1);
			data_type beta1 = get_beta(clear_acts_diff, 0);
			data_type beta2 = get_beta(clear_acts_diff, 1);

			auto &input1 = *inputs_[0]->get(device_index);
			auto &input2 = *inputs_[1]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);

			KERNEL_CALL(eltmax_backward, input1.count())(input1.count(), input1.gpu_data(), input2.gpu_data(), output.gpu_diff(), input1.mutable_gpu_diff(), input2.mutable_gpu_diff(),
				beta1, beta2, bp1, bp2);

			CUDA_POST_KERNEL_CHECK;
		}
	}
}