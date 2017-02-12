
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class resize_grid_layer : public caffepro_layer {
	public:
		resize_grid_layer(caffepro_context *context, const LayerParameter &param);
		~resize_grid_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		void get_output_size(int device_index, int &output_width, int &output_height);

	protected:
		boost::shared_ptr<node_blob> buffer_, sum_multiplier_;
		data_type output_box_start_, output_box_scale_;
		int output_min_length_;
		data_type output_max_scale_;
	};
}