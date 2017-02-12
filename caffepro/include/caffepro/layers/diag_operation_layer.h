
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class diag_operation_layer : public caffepro_layer {
	public:
		diag_operation_layer(caffepro_context *context, const LayerParameter &param);
		~diag_operation_layer();

	public:
		virtual void init();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		data_type scale_, shift_;
	};
}