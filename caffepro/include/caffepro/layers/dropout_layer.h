
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class dropout_layer : public caffepro_layer {
	public:
		dropout_layer(caffepro_context *context, const LayerParameter &param);
		~dropout_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		data_type threshold_, scale_;
		unsigned int uint_thres_;
		bool force_random_;
		bool force_same_;

		boost::shared_ptr<node_blob> rand_vec_;
	};
}