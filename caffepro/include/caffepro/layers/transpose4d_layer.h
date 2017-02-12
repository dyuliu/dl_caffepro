
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class transpose4d_layer : public caffepro_layer {
	public:
		transpose4d_layer(caffepro_context *context, const LayerParameter &param);
		~transpose4d_layer();

	public:
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		int width_, height_, channels_in_, channels_out_; // original dims
		int tr_width_, tr_height_, tr_channels_in_, tr_channels_out_; // transposed dims
	};
}