
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {

	class sigmoid_layer;

	class multilabel_sigmoid_loss_layer : public caffepro_layer {
	public:
		multilabel_sigmoid_loss_layer(caffepro_context *context, const LayerParameter &param);
		~multilabel_sigmoid_loss_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	public:
		// fetch functions
		const boost::shared_ptr<node_blob> &prob() const { return prob_; }
		const boost::shared_ptr<node_blob> &avg_prob() const { return avg_prob_; }

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		boost::shared_ptr<sigmoid_layer> sigmoid_;

		boost::shared_ptr<node_blob> prob_, correct_, sum_multiplier_, avg_prob_;
		layer_io_buffer sigmoid_inputs_, sigmoid_outputs_;

		data_type coeff_;
	};
}