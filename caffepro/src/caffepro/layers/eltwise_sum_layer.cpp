
#include <caffepro/layers/eltwise_sum_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

#include <opencv2/opencv.hpp>


namespace caffepro {
	eltwise_sum_layer::eltwise_sum_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = 1;
		attr_.num_inputs_max = INT_MAX;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_SAME_COUNT
			| layer_attribute::CF_REQUIRE_SAME_NUM
			);
	}

	eltwise_sum_layer::~eltwise_sum_layer() {
		release_all();
	}

	void eltwise_sum_layer::init() {
		check_input();

		int coeff_size = (int)layer_param_.eltwise_sum_param().coeff_size();
		CHECK(coeff_size == 0 || coeff_size == (int)inputs_.size());
	
		coeff_.clear();
		if (coeff_size == 0) {
			for (int i = 0; i < (int)inputs_.size(); i++) {
				coeff_.push_back((data_type)1.f);
			}
		}
		else {
			for (int i = 0; i < coeff_size; i++) {
				coeff_.push_back(layer_param_.eltwise_sum_param().coeff(i));
			}
		}
	}

	void eltwise_sum_layer::on_forward(int device_index) {
		const int count = outputs_[0]->get(device_index)->count();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();

		// inplace friendly
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		cublas.scale(count, coeff_[0], inputs_[0]->get(device_index)->gpu_data(), top_data);
		for (int i = 1; i < inputs_.size(); ++i) {
			cublas.axpy(count, coeff_[i], inputs_[i]->get(device_index)->gpu_data(), top_data);
		}

		return;
	}

	void eltwise_sum_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const int count = outputs_[0]->get(device_index)->count();
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		for (int i = (int)inputs_.size() - 1; i >= 0; --i) { // inplace friendly
			if (should_bp(bp_acts, i)) {
				const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
				data_type* bottom_diff = inputs_[i]->get(device_index)->mutable_gpu_diff();

				const data_type beta_acts = get_beta(clear_acts_diff, i);

				if (beta_acts == 0) {
					cublas.scale(count, coeff_[i], top_diff, bottom_diff);
				}
				else {
					cublas.axpby(count, coeff_[i], top_diff, beta_acts, bottom_diff);
				}
			}
		}
	}
}