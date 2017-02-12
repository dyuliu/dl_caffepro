
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class conv_layer;

	class correlation_layer : public caffepro_layer {
	public:
		correlation_layer(caffepro_context *context, const LayerParameter &param);
		~correlation_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		// override forward and backward directly
		// do not use on_forward or on_backward
		virtual void forward();
		virtual void backward(act_selector bp_acts = UINT32_MAX, weight_selector bp_weights = UINT32_MAX,
			act_selector clear_acts_diff = UINT32_MAX, weight_selector clear_weights_diff = UINT32_MAX);
	
	protected:
		// do nothing
		virtual void on_forward(int device_index) {}
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {}

	protected:
		boost::shared_ptr<conv_layer> conv_;
		boost::shared_ptr<LayerParameter> conv_parameter_;
		layer_io_buffer conv_inputs_, conv_outputs_;
		int channels_;
	};
}