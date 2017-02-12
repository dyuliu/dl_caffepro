
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class dropoutsame_layer : public caffepro_layer {
	public:
		dropoutsame_layer(caffepro_context *context, const LayerParameter &param);
		~dropoutsame_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_before_forward();
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		data_type threshold_, scale_;
		unsigned int uint_thres_;
		bool force_random_;

		bool open_or_not_;
	};
}