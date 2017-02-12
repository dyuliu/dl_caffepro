
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class sample_layer : public caffepro_layer {
	public:
		sample_layer(caffepro_context *context, const LayerParameter &param);
		~sample_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		void get_grid_backtrack_index(int device_index, std::vector<int> &indexes, std::vector<int> &index_start);

	protected:
		boost::shared_ptr<node_blob> grid_buffer_, grid_sum_multiplier_;
		boost::shared_ptr<node_blob> act_buffer_, act_sum_multiplier_;
	};
}