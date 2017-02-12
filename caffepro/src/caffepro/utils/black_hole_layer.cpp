
#include <caffepro/layers/black_hole_layer.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	black_hole_layer::black_hole_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = 1;
		attr_.num_inputs_max = INT_MAX;
		attr_.num_outputs_min = attr_.num_outputs_max = 0;
		attr_.set_constraint(layer_attribute::CF_REQUIRE_SAME_DEVICE);

		attr_.device_dispatcher_forward = layer_attribute::INPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::INPUT_BASE;
	}

	black_hole_layer::~black_hole_layer() {
		release_all();
	}

	void black_hole_layer::resize() {
		check_input();
	}

	void black_hole_layer::on_forward(int device_index) {
		// do nothing
	}

	void black_hole_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		for (int i = 0; i < (int)inputs_.size(); i++) {
			if (should_bp(bp_acts, i) && get_beta(clear_acts_diff, i) == 0) {
				inputs_[i]->get(device_index)->fill_diff(0.f);
			}
		}
	}
}