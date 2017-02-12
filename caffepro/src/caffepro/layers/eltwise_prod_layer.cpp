
#include <caffepro/layers/eltwise_prod_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	eltwise_prod_layer::eltwise_prod_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = 1;
		attr_.num_inputs_max = INT_MAX;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			//| layer_attribute::CF_ALLOW_INPLACE
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_SAME_COUNT
			| layer_attribute::CF_REQUIRE_SAME_NUM
			);
	}

	eltwise_prod_layer::~eltwise_prod_layer() {
		release_all();
	}

	void eltwise_prod_layer::init() {
		check_input();
	}

	void eltwise_prod_layer::on_forward(int device_index) {
		const int count = outputs_[0]->get(device_index)->count();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		cublas.copy(count, inputs_[0]->get(device_index)->gpu_data(), top_data);
		for (int i = 1; i < inputs_.size(); ++i) {
			cublas.mul(count, top_data, inputs_[i]->get(device_index)->gpu_data(), top_data);
		}
	}

	void eltwise_prod_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const int count = outputs_[0]->get(device_index)->count();
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		for (int i = (int)inputs_.size() - 1; i >= 0; --i) { 
			if (should_bp(bp_acts, i)) {
				const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
				data_type* bottom_diff = inputs_[i]->get(device_index)->mutable_gpu_diff();

				const data_type beta_acts = get_beta(clear_acts_diff, i);
				
				//CHECK_EQ(beta_acts, 0);

				if (beta_acts == 0) {
					cublas.copy(count, top_diff, bottom_diff);
					for (int j = (int)inputs_.size() - 1; j >= 0; --j) {
						if (j == i) continue;
						cublas.mul(count, bottom_diff, inputs_[j]->get(device_index)->gpu_data(), bottom_diff);
					}//j
				}
				else {
					data_type *buffer = (data_type*)context_->get_current_device()->memory()->allocate(count * sizeof(data_type));					
					cublas.copy(count, top_diff, buffer);
					for (int j = (int)inputs_.size() - 1; j >= 0; --j) {
						if (j == i) continue;
						cublas.mul(count, buffer, inputs_[j]->get(device_index)->gpu_data(), buffer);
					}//j
					cublas.axpby(count, 1.0, buffer, beta_acts, bottom_diff);
					context_->get_current_device()->memory()->free(buffer);
				}// if beta_acts
			}
		}//i
	}
}