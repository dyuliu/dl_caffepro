
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class eltwise_max_layer : public caffepro_layer {
	public:
		eltwise_max_layer(caffepro_context *context, const LayerParameter &param);
		~eltwise_max_layer();

	public:
		// overrides
		virtual void init();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
	};
}