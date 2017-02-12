
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class mean_norm_layer : public caffepro_layer {
	public:
		mean_norm_layer(caffepro_context *context, const LayerParameter &param);
		~mean_norm_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_after_forward();

	protected:
		int channels_;
		int record_iters_;
		int batch_size_;

		boost::shared_ptr<node_blob> EX_, EX_batch_, sum_multiplier_, sum_multiplier_num_;
	};
}