
#pragma once

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	
	class softmax_layer;

	class softmax_loss_ohem_layer : public caffepro_layer {
	public:
		softmax_loss_ohem_layer(caffepro_context *context, const LayerParameter &param);
		~softmax_loss_ohem_layer();

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
		boost::shared_ptr<softmax_layer> softmax_;
		
		boost::shared_ptr<node_blob> prob_, correct_, sum_multiplier_, avg_prob_;
		boost::shared_ptr<node_blob> bp_indicator_;

		layer_io_buffer softmax_inputs_, softmax_outputs_;

		data_type coeff_;
		int output_top_n_;

		int ohem_size_;
		bool force_random_;
	};
}