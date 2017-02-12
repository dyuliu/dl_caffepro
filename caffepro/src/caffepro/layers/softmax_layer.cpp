
#include <caffepro/layers/softmax_layer.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	softmax_layer::softmax_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	softmax_layer::~softmax_layer() {
		release_all();
	}

	void softmax_layer::init() {
		check_input();

		init_cudnn();
	}

	void softmax_layer::init_cudnn() {
		int n_devices = (int)inputs_[0]->size();
		softmax_contexts_.resize(n_devices);

		for (int nd = 0; nd < n_devices; nd++) {
			memset(&softmax_contexts_[nd], 0, sizeof(cudnn_softmax_context));

			CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmax_contexts_[nd].bottom_desc));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmax_contexts_[nd].top_desc));
		}
	}

	void softmax_layer::resize() {
		caffepro_layer::resize();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				resize_cudnn(nd);
			}
		}
	}

	void softmax_layer::resize_cudnn(int device_index) {
		int num = inputs_[0]->get(device_index)->num();
		int width = inputs_[0]->get(device_index)->width();
		int height = inputs_[0]->get(device_index)->height();
		int channels = inputs_[0]->get(device_index)->channels();

		ENTER_DEVICE_CONTEXT(inputs_[0]->get(device_index)->device_id())
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				softmax_contexts_[device_index].bottom_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				num,
				channels,
				height,
				width
				));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				softmax_contexts_[device_index].top_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				num,
				channels,
				height,
				width
				));
		EXIT_DEVICE_CONTEXT
	}

	void softmax_layer::release_all() {
		for (auto pDes = softmax_contexts_.begin(); pDes != softmax_contexts_.end(); ++pDes) {
			if (pDes->bottom_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->bottom_desc));
			}
			if (pDes->top_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->top_desc));
			}
		}
	}

	void softmax_layer::on_forward(int device_index) {
		const data_type alpha = 1.f, beta = 0.f;
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();

		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		
		CUDNN_CHECK(cudnnSoftmaxForward(
			handle, 
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			softmax_contexts_[device_index].bottom_desc, 
			bottom_data,
			&beta,
			softmax_contexts_[device_index].top_desc,
			top_data
			));
	}

	void softmax_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type alpha = 1;
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();

		if (should_bp(bp_acts, 0)) {

			const data_type* top_data = outputs_[0]->get(device_index)->gpu_data();
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();

			CUDNN_CHECK(cudnnSoftmaxBackward(
				handle,
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				softmax_contexts_[device_index].top_desc,
				top_data,
				softmax_contexts_[device_index].top_desc,
				top_diff,
				&beta_acts,
				softmax_contexts_[device_index].bottom_desc,
				bottom_diff
				));
		}
	}
}