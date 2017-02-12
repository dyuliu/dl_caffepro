
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class gpu_concat_layer : public caffepro_layer {
	public:
		gpu_concat_layer(caffepro_context *context, const LayerParameter &param);
		~gpu_concat_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_after_forward();
		virtual void on_before_backward();
		virtual void on_after_backward();

	protected:
		int concated_device_id_;
		std::vector<int> nd_offsets_;
	};
}