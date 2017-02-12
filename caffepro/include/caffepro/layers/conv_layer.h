
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class conv_layer : public caffepro_layer {
	public:
		conv_layer(caffepro_context *context, const LayerParameter &param);
		~conv_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();
		virtual void release_all();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	protected:
		void init_cudnn();
		void resize_cudnn(int device_index);
		void find_fastest_cudnn_algo(int device_index);

	protected:
		int channels_, num_outputs_;
		int ksize_w_, ksize_h_, stride_w_, stride_h_, pad_w_, pad_h_;
		bool bias_term_;
		bool size_floor_;

	protected:
		// cudnn handles
		struct cudnn_conv_context {
			cudnnConvolutionFwdAlgo_t fwd_algo;
			cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
			cudnnConvolutionBwdDataAlgo_t bwd_data_algo;

			cudnnFilterDescriptor_t filter_desc;
			cudnnTensorDescriptor_t bottom_desc, top_desc;
			cudnnTensorDescriptor_t bias_desc;
			cudnnConvolutionDescriptor_t conv_desc;
		};

		std::vector<cudnn_conv_context> conv_contexts_;
		boost::shared_ptr<node_blob> workspace_;
	};
};