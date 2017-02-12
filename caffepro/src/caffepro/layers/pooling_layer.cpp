
#include <caffepro/layers/pooling_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	pooling_layer::pooling_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	pooling_layer::~pooling_layer() {
		release_all();
	}

	void pooling_layer::init() {
		check_input();

		auto &pool_param = layer_param_.pool_param();

		if (pool_param.has_kernel_size_x() || pool_param.has_kernel_size_y()) {
			CHECK(!pool_param.has_kernel_size());
			ksize_w_ = pool_param.kernel_size_x();
			ksize_h_ = pool_param.kernel_size_y();
		}
		else {
			ksize_w_ = pool_param.kernel_size();
			ksize_h_ = ksize_w_; 
		}
		if (pool_param.has_stride_x() || pool_param.has_stride_y()) {
			CHECK(!pool_param.has_stride());
			stride_w_ = pool_param.stride_x();
			stride_h_ = pool_param.stride_y();
		}
		else {
			stride_w_ = pool_param.stride();
			stride_h_ = stride_w_;
		}
		if (pool_param.has_pad_x() || pool_param.has_pad_y()) {
			CHECK(!pool_param.has_pad());
			pad_w_ = pool_param.pad_x();
			pad_h_ = pool_param.pad_y();
		}
		else {
			pad_w_ = pool_param.pad();
			pad_h_ = pad_w_; 
		}
		size_floor_ = layer_param_.pool_param().size_floor();

		init_cudnn();
	}

	void pooling_layer::init_cudnn() {
		int n_devices = (int)inputs_[0]->size();
		pool_contexts_.resize(n_devices);

		cudnnPoolingMode_t mode;
		if (layer_param_.pool_param().pool() == PoolingParameter_PoolMethod_MAX) {
			mode = CUDNN_POOLING_MAX;
		}
		else if (layer_param_.pool_param().pool() == PoolingParameter_PoolMethod_AVE) {
			mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		}
		else {
			LOG(FATAL) << "Unknown pooling type";
		}

		for (int nd = 0; nd < n_devices; nd++) {
			memset(&pool_contexts_[nd], 0, sizeof(cudnn_pool_context));
		
			ENTER_DEVICE_CONTEXT(inputs_[0]->get(nd)->device_id())
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool_contexts_[nd].bottom_desc));
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&pool_contexts_[nd].top_desc));
				CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_contexts_[nd].pooling_desc));
				CUDNN_CHECK(cudnnSetPooling2dDescriptor(
					pool_contexts_[nd].pooling_desc,
					mode,
					CUDNN_PROPAGATE_NAN,
					ksize_h_,
					ksize_w_,
					pad_h_,
					pad_w_,
					stride_h_,
					stride_w_
					));
			EXIT_DEVICE_CONTEXT;
		}
	}

	void pooling_layer::resize() {
		check_input();

		int n_devices = (int)inputs_[0]->size();

		if (outputs_[0]->size() == 0) {
			for (int nd = 0; nd < n_devices; nd++) {
				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_like_4dext(
					inputs_[0]->get(nd)->context(),
					*inputs_[0]->get(nd),
					inputs_[0]->get(nd)->channels(),
					ksize_h_,
					ksize_w_,
					stride_h_,
					stride_w_,
					pad_h_,
					pad_w_,
					size_floor_,
					inputs_[0]->get(nd)->device_id()
					)
					));

				resize_cudnn(nd);
			}
		}
		else {
			CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());

			for (int nd = 0; nd < n_devices; nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					outputs_[0]->get(nd)->reshape_like_4dext(
						*inputs_[0]->get(nd),
						inputs_[0]->get(nd)->channels(),
						ksize_h_,
						ksize_w_,
						stride_h_,
						stride_w_,
						pad_h_,
						pad_w_,
						size_floor_
						);

					resize_cudnn(nd);
				}
			}
		}
	}

	void pooling_layer::resize_cudnn(int device_index) {
		int num = inputs_[0]->get(device_index)->num();
		int width = inputs_[0]->get(device_index)->width();
		int height = inputs_[0]->get(device_index)->height();
		int channels = inputs_[0]->get(device_index)->channels();
		int out_width = outputs_[0]->get(device_index)->width();
		int out_height = outputs_[0]->get(device_index)->height();

		cudnnHandle_t handle = context_->get_device(inputs_[0]->get(device_index)->device_id())->cudnn_handle();

		ENTER_DEVICE_CONTEXT(inputs_[0]->get(device_index)->device_id())
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				pool_contexts_[device_index].bottom_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				num,
				channels,
				height,
				width
				));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				pool_contexts_[device_index].top_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				num,
				channels,
				out_height,
				out_width
				));
		EXIT_DEVICE_CONTEXT
	}

	void pooling_layer::release_all() {
		for (auto pDes = pool_contexts_.begin(); pDes != pool_contexts_.end(); ++pDes) {
			if (pDes->bottom_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->bottom_desc));
			}
			if (pDes->top_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->top_desc));
			}
			if (pDes->pooling_desc) {
				CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pDes->pooling_desc));
			}
		}
	}

	void pooling_layer::on_forward(int device_index) {
		const data_type alpha = 1.f, beta = 0.f;
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();

		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		CUDNN_CHECK(cudnnPoolingForward(
			handle, 
			pool_contexts_[device_index].pooling_desc,
			&alpha,
			pool_contexts_[device_index].bottom_desc,
			bottom_data,
			&beta,
			pool_contexts_[device_index].top_desc, 
			top_data
			));
	}

	void pooling_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type alpha = 1;
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();
		
		if (should_bp(bp_acts, 0)) {
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			const data_type* top_data = outputs_[0]->get(device_index)->gpu_data();
			const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();

			CUDNN_CHECK(cudnnPoolingBackward(
				handle, 
				pool_contexts_[device_index].pooling_desc,
				&alpha,
				pool_contexts_[device_index].top_desc,
				top_data, 
				pool_contexts_[device_index].top_desc, 
				top_diff,
				pool_contexts_[device_index].bottom_desc, 
				bottom_data,
				&beta_acts,
				pool_contexts_[device_index].bottom_desc, 
				bottom_diff
				));
		}
	}
}