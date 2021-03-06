
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class step_gate_layer : public caffepro_layer {
	public:
		step_gate_layer(caffepro_context *context, const LayerParameter &param);
		~step_gate_layer();

	public:
		virtual void init();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		data_type init_value_, step_value_, max_value_;
	};
}