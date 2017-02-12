
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class black_hole_layer : public caffepro_layer {
	public:
		black_hole_layer(caffepro_context *context, const LayerParameter &param);
		~black_hole_layer();

	public:
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
	};
}