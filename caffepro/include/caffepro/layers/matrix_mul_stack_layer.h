
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class matrix_mul_layer;
	class LayerParameter;

	class matrix_mul_stack_layer : public caffepro_layer {
	public:
		matrix_mul_stack_layer(caffepro_context *context, const LayerParameter &param);
		~matrix_mul_stack_layer();

	public:
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void forward();
		virtual void backward(act_selector bp_acts = UINT32_MAX, weight_selector bp_weights = UINT32_MAX,
			act_selector clear_acts_diff = UINT32_MAX, weight_selector clear_weights_diff = UINT32_MAX);

		virtual void on_forward(int device_index) {}
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {}

	protected:
		std::vector<boost::shared_ptr<matrix_mul_layer> > mul_layers_;
		std::vector<boost::shared_ptr<LayerParameter> > mul_params_;
		std::vector<layer_io_buffer> mul_inputs_, mul_outputs_;
		int n_terms_;
	};
}