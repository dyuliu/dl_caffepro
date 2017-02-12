
#include <caffepro/layers/rand_select_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>


namespace caffepro {
	rand_select_layer::rand_select_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = 1;
		attr_.num_inputs_max = INT_MAX;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_COUNT
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			);
	}

	rand_select_layer::~rand_select_layer() {
		release_all();
	}

	void rand_select_layer::init() {
		check_input();

		selected_ = 0;
	}

	void rand_select_layer::on_before_forward() {
		if (context_->get_phase() == caffepro_context::TRAIN) {
			selected_ = rand() % inputs_.size();
		}
	}

	void rand_select_layer::on_forward(int device_index) {
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		int count = outputs_[0]->get(device_index)->count();

		if (context_->get_phase() == caffepro_context::TRAIN) {
			cublas.copy(count, inputs_[selected_]->get(device_index)->gpu_data(), outputs_[0]->get(device_index)->mutable_gpu_data());
		}
		else {
			data_type scal = 1.f / inputs_.size();
			cublas.scale(count, scal, inputs_[0]->get(device_index)->gpu_data(), outputs_[0]->get(device_index)->mutable_gpu_data());
			for (int i = 1; i < (int)inputs_.size(); i++) {
				cublas.axpy(count, scal, inputs_[i]->get(device_index)->gpu_data(), outputs_[0]->get(device_index)->mutable_gpu_data());
			}
		}
	}

	void rand_select_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		int count = outputs_[0]->get(device_index)->count();

		for (int i = 0; i < (int)inputs_.size(); i++) {
			if (should_bp(bp_acts, i)) {
				data_type beta = get_beta(bp_acts, i);
				if (i == selected_) {
					if (beta == 0) {
						cublas.copy(count, outputs_[0]->get(device_index)->gpu_diff(), inputs_[i]->get(device_index)->mutable_gpu_diff());
					}
					else {
						cublas.axpby(count, 1.f, outputs_[0]->get(device_index)->gpu_diff(), beta, inputs_[i]->get(device_index)->mutable_gpu_diff());
					}
				}
				else {
					if (beta == 0) {
						inputs_[i]->get(device_index)->fill_diff(0.f);
					}
				}
			}
		}
	}
}
