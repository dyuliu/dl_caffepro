
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class cluster_loss_layer : public caffepro_layer {
	public:
		cluster_loss_layer(caffepro_context *context, const LayerParameter &param);
		~cluster_loss_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_before_forward();
		virtual void on_after_forward();

	protected:
		data_type coeff_;
		int num_centers_, num_dims_;
		bool reset_centers_;

		boost::shared_ptr<node_blob> distance_matrix_, assign_matrix_, assign_matrix_back_, loss_matrix_;
	};
}