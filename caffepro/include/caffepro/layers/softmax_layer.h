
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>

namespace caffepro {
	class softmax_layer : public caffepro_layer {
	public:
		softmax_layer(caffepro_context *context, const LayerParameter &param);
		~softmax_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();
		virtual void release_all();

	protected:
		void init_cudnn();
		void resize_cudnn(int device_index);

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
	
	protected:
		// cudnn handles
		struct cudnn_softmax_context {
			cudnnTensorDescriptor_t bottom_desc, top_desc;
		};

		std::vector<cudnn_softmax_context> softmax_contexts_;
	};
}