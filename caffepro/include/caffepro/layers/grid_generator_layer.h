
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class grid_generator_layer : public caffepro_layer {
	public:
		grid_generator_layer(caffepro_context *context, const LayerParameter &param);
		~grid_generator_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		std::string method_;
		data_type scale_w_, scale_h_;

		boost::shared_ptr<node_blob> buffer_, sum_multiplier_; 
	};
}