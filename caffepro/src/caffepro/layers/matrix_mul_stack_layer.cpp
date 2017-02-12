
#include <caffepro/layers/matrix_mul_stack_layer.h>
#include <caffepro/layers/matrix_mul_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	matrix_mul_stack_layer::matrix_mul_stack_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = 2;
		attr_.num_inputs_max = INT_MAX;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_SAME_COUNT_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_INNER_COUNT_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	matrix_mul_stack_layer::~matrix_mul_stack_layer() {
		release_all();
	}

	void matrix_mul_stack_layer::init() {
		check_input();

		n_terms_ = (int)inputs_.size() - 1;

		for (int i = 0; i < n_terms_; i++) {
			bool trans_A = layer_param_.matrix_mul_stack_param().trans_odd();
			bool trans_B = layer_param_.matrix_mul_stack_param().trans_even();

			if (i & 1) {
				std::swap(trans_A, trans_B);
			}

			mul_params_.push_back(boost::shared_ptr<LayerParameter>(new LayerParameter()));
			mul_params_.back()->mutable_matrix_mul_param()->set_trans_a(trans_A);
			mul_params_.back()->mutable_matrix_mul_param()->set_trans_b(trans_B);
			mul_layers_.push_back(boost::shared_ptr<matrix_mul_layer>(new matrix_mul_layer(context_, *mul_params_.back())));
			
			layer_io_buffer input, output;
			if (i == 0) {
				input.push_back(inputs_[0]);
				input.push_back(inputs_[1]);
			}
			else {
				input.push_back(mul_outputs_[i - 1][0]);
				input.push_back(inputs_[i + 1]);
			}
			if (i != n_terms_ - 1) {
				output.push_back(boost::shared_ptr<node_blob>(new node_blob()));
			}
			else {
				output.push_back(outputs_[0]);
			}

			mul_inputs_.push_back(input);
			mul_outputs_.push_back(output);

			mul_layers_.back()->bind(mul_inputs_.back(), mul_outputs_.back());
			mul_layers_.back()->init();
			mul_layers_.back()->resize();
		}
	}

	void matrix_mul_stack_layer::resize() {
		check_input();

		bool reshaped = false;
		for (auto &input : inputs_) {
			if (input->reshaped()) {
				reshaped = true;
				break;
			}
		}

		if (reshaped) {
			for (auto &mul_layer : mul_layers_) {
				mul_layer->resize();
			}
		}
	}

	void matrix_mul_stack_layer::forward() {
		for (auto &mul_layer : mul_layers_) {
			mul_layer->forward();
		}
	}

	void matrix_mul_stack_layer::backward(act_selector bp_acts, weight_selector bp_weights,
		act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		for (int i = n_terms_ - 1; i >= 0; i--) {
			act_selector bp = 1, clear = 1;
			if (i == 0) {
				bp = should_bp(bp_acts, 0) ? 1 : 0;
				clear = get_beta(clear_acts_diff, 0) == 0 ? 1 : 0;
			}
			bp |= should_bp(bp_acts, i + 1) ? 2 : 0;
			clear |= get_beta(clear_acts_diff, i + 1) == 0 ? 2 : 0;

			mul_layers_[i]->backward(bp, 0, clear, 0);
		}
	}
}