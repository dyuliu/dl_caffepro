
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {

	class softmax_layer;
	class gpu_concat_layer;

	class rcnn_loss_layer : public caffepro_layer {
	public:
		rcnn_loss_layer(caffepro_context *context, const LayerParameter &param);
		~rcnn_loss_layer();

	public:
		// fetch functions
		const boost::shared_ptr<node_blob> &prob() const { return prob_; }
		const boost::shared_ptr<node_blob> &avg_prob() const { return avg_prob_dev_; }
		const boost::shared_ptr<node_blob> &correct() const { return correct_; }

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_before_forward();
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_after_forward();

	protected:
		boost::shared_ptr<softmax_layer> softmax_;

		layer_io_buffer softmax_inputs_, softmax_outputs_;
		boost::shared_ptr<node_blob> prob_, correct_, sum_multiplier_, avg_prob_dev_;

		data_type coeff_;
	};

}