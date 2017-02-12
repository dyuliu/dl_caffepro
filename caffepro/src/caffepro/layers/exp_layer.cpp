
#include <caffepro/layers/exp_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	exp_layer::exp_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS
			);
	}

	exp_layer::~exp_layer() {
		release_all();
	}

	void exp_layer::init() {
		check_input();

		slope_ = layer_param_.exp_param().slope();
	}

	void exp_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		cublas.scale(count, slope_, bottom_data, top_data);
		cublas.exp(count, top_data, top_data);
	}

	void exp_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* top_data = outputs_[0]->get(device_index)->gpu_data(); // use top data here, not bottom data
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const int count = inputs_[0]->get(device_index)->count();
			CHECK_EQ(beta_acts, 0);
			
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.mul(count, top_diff, top_data, bottom_diff);
			cublas.scal(count, slope_, bottom_diff);
		}
	}
}