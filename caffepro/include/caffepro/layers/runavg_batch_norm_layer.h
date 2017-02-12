
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>


namespace caffepro {
	class runavg_batch_norm_layer : public caffepro_layer {
	public:
		runavg_batch_norm_layer(caffepro_context *context, const LayerParameter &param);
		~runavg_batch_norm_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();
		virtual void release_all();

	protected:
		void init_cudnn();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_after_forward();

	protected:
		double eps_, running_factor_;
		int record_iters_;
		int channels_;
		boost::shared_ptr<node_blob> EX_batch_, VarX_batch_;

		// for cudnn
		std::vector<cudnnTensorDescriptor_t> feature_desc_, mean_var_desc_;
	};
}