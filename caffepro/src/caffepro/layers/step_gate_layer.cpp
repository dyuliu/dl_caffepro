
#include <caffepro/layers/step_gate_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <boost/scoped_ptr.hpp>

namespace caffepro {
	step_gate_layer::step_gate_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			);
	}

	step_gate_layer::~step_gate_layer() {
		release_all();
	}

	void step_gate_layer::init() {
		check_input();

		init_value_ = layer_param_.step_gate_param().init_value();
		step_value_ = layer_param_.step_gate_param().step_value(); 
		max_value_ = layer_param_.step_gate_param().max_value();

		weights_.push_back(boost::shared_ptr<node_blob>(new node_blob()));
		boost::scoped_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, 1, 1, 1, -1));
		weights_[0]->add_like(*weight_template, *inputs_[0]);

		weights_[0]->get(0)->fill_data(init_value_);
		weights_[0]->broadcast_data_via_gpu(0);
		weights_[0]->get(0)->fill_diff(0.f);
		weights_[0]->broadcast_diff_via_gpu(0);

		if (layer_param_.step_gate_param().has_start_iter()) {
			weights_.push_back(boost::shared_ptr<node_blob>(new node_blob()));
			weights_[1]->add_like(*weight_template, *inputs_[0]);
			weights_[1]->get(0)->fill_data(0.f);
			weights_[1]->get(0)->fill_diff(0.f);
			weights_[1]->broadcast_data_via_cpu(0); // use cpu here
			weights_[1]->broadcast_diff_via_gpu(0);
		}
	}

	void step_gate_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		if (!layer_param_.step_gate_param().keep_forward()) {
			cublas.scale_dev(count, weights_[0]->get(device_index)->gpu_data(), bottom_data, top_data);
		}
		else {
			cublas.copy(count, bottom_data, top_data);
		}
	}

	void step_gate_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const int count = inputs_[0]->get(device_index)->count();
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

			if (!layer_param_.step_gate_param().keep_backward()) { // normal
				if (outputs_[0] == inputs_[0]) { // inplace
					cublas.scale_dev(count, weights_[0]->get(device_index)->gpu_data(), top_diff, bottom_diff);
				}
				else {
					const data_type beta_acts = get_beta(clear_acts_diff, 0);
					cublas.axpby_dev(count, weights_[0]->get(device_index)->gpu_data(), top_diff, beta_acts, bottom_diff);
				}
			}
			else { // keep_backward
				if (outputs_[0] == inputs_[0]) { // inplace
					// do nothing
				}
				else {
					const data_type beta_acts = get_beta(clear_acts_diff, 0);
					cublas.axpby(count, (data_type)1.f, top_diff, beta_acts, bottom_diff);
				}
			}

			// update weights
			bool update = true;
			if (layer_param_.step_gate_param().has_start_iter()) {
				data_type *iter_data = weights_[1]->get(device_index)->mutable_cpu_data();
				if (*iter_data < layer_param_.step_gate_param().start_iter()) {
					update = false;
				}
				(*iter_data) += 1.f;
				weights_[1]->get(device_index)->fill_diff(0.f);
			}
			if (update) {
				cublas.add_scalar(1, step_value_, weights_[0]->get(device_index)->mutable_gpu_data());
				if (step_value_ > 0) {
					cublas.min_scalar(1, max_value_, weights_[0]->get(device_index)->mutable_gpu_data());
				}
				else if (step_value_ < 0) {
					cublas.max_scalar(1, max_value_, weights_[0]->get(device_index)->mutable_gpu_data());
				}
			}

			// clear diff
			weights_[0]->get(device_index)->fill_diff(0.f);
		}
	}
}