
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class sigmoid_layer;

	class learnable_dropout_layer : public caffepro_layer {
	public:
		learnable_dropout_layer(caffepro_context *context, const LayerParameter &param);
		~learnable_dropout_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_before_forward();
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void backward(act_selector bp_acts = UINT32_MAX, weight_selector bp_weights = UINT32_MAX,
			act_selector clear_acts_diff = UINT32_MAX, weight_selector clear_weights_diff = UINT32_MAX);

	protected:
		boost::shared_ptr<node_blob> rand_vec_, thres_;
		layer_io_buffer sigmoid_inputs_, sigmoid_outputs_;
		boost::shared_ptr<sigmoid_layer> sigmoid_;
	};
}