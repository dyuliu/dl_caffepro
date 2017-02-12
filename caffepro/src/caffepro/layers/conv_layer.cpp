
#include <caffepro/layers/conv_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/utils/utils.h>
#include <caffepro/context/common_names.h>
#include <boost/scoped_ptr.hpp>

#include <caffepro/math/cublas_debug.h>

namespace caffepro {
	conv_layer::conv_layer(caffepro_context *context, const LayerParameter &param) 
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	conv_layer::~conv_layer() {
		release_all();
	}

	void conv_layer::init() {
		check_input();

		auto &conv_param = layer_param_.conv_param();

		if (conv_param.has_kernel_size_x() || conv_param.has_kernel_size_y()) {
			CHECK(!conv_param.has_kernel_size());
			ksize_w_ = conv_param.kernel_size_x();
			ksize_h_ = conv_param.kernel_size_y();
		}
		else {
			ksize_w_ = conv_param.kernel_size();
			ksize_h_ = ksize_w_;
		}
		if (conv_param.has_stride_x() || conv_param.has_stride_y()) {
			CHECK(!conv_param.has_stride());
			stride_w_ = conv_param.stride_x();
			stride_h_ = conv_param.stride_y();
		}
		else {
			stride_w_ = conv_param.stride();
			stride_h_ = stride_w_; // currently supported
		}
		if (conv_param.has_pad_x() || conv_param.has_pad_y()) {
			CHECK(!conv_param.has_pad());
			pad_w_ = conv_param.pad_x();
			pad_h_ = conv_param.pad_y();
		}
		else {
			pad_w_ = conv_param.pad();
			pad_h_ = pad_w_; // currently supported
		}
		bias_term_ = layer_param_.conv_param().bias_term();
		size_floor_ = layer_param_.conv_param().size_floor();
		num_outputs_ = layer_param_.conv_param().num_output();
		
		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		channels_ = inputs_[0]->dim_at(2);

		internal_weights_.clear();

		// set weights
		if (bias_term_) {
			weights_.resize(2);
		}
		else {
			weights_.resize(1);
		}

		int n_devices = (int)inputs_[0]->size();
		boost::scoped_ptr<device_blob> weight_template(
			device_blob::create_4d(context_, num_outputs_, channels_, ksize_h_, ksize_w_, inputs_[0]->get(0)->device_id()));
		weights_[0].reset(new node_blob());
		weights_[0]->add_like(context_, *weight_template, *inputs_[0]);
		boost::scoped_ptr<filler> weight_filler(filler::create(context_, layer_param_.conv_param().weight_filler()));
		weight_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		if (bias_term_) {
			boost::scoped_ptr<device_blob> bias_template(
				device_blob::create_4d(context_, 1, num_outputs_, 1, 1, inputs_[0]->get(0)->device_id())
				);
			weights_[1].reset(new node_blob());
			weights_[1]->add_like(context_, *bias_template, *inputs_[0]);
			boost::scoped_ptr<filler> bias_filler(filler::create(context_, layer_param_.conv_param().bias_filler()));
			bias_filler->fill(*weights_[1]->get(0));
			weights_[1]->broadcast_data_via_gpu(0);
		}

		init_cudnn();
	}

	void conv_layer::init_cudnn() {
		int n_devices = (int)inputs_[0]->size();
		conv_contexts_.resize(n_devices);
		workspace_.reset(new node_blob());
		workspace_->set_attr(node_blob::NF_TEMP);
		internal_weights_.push_back(workspace_);

		for (int nd = 0; nd < n_devices; nd++) {
			memset(&conv_contexts_[nd], 0, sizeof(cudnn_conv_context));

			ENTER_DEVICE_CONTEXT(inputs_[0]->get(nd)->device_id())
				CUDNN_CHECK(cudnnCreateFilterDescriptor(&conv_contexts_[nd].filter_desc));
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv_contexts_[nd].bottom_desc));
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv_contexts_[nd].top_desc));
				CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_contexts_[nd].conv_desc));
				if (bias_term_) {
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&conv_contexts_[nd].bias_desc));
				}
				workspace_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, 1, 1, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				CUDNN_CHECK(cudnnSetFilter4dDescriptor(
					conv_contexts_[nd].filter_desc,
					CUDNN_DATA_FLOAT,
					CUDNN_TENSOR_NCHW,
					num_outputs_,
					channels_,
					ksize_h_,
					ksize_w_
					));

			EXIT_DEVICE_CONTEXT;
		}
	}

	void conv_layer::resize() {
		check_input();
		int n_devices = (int)inputs_[0]->size();

		if (outputs_[0]->size() == 0) {
			for (int nd = 0; nd < n_devices; nd++) {
				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_like_4dext(
						inputs_[0]->get(nd)->context(),
						*inputs_[0]->get(nd),
						num_outputs_,
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

				if (context_->get_global_cfg(GLOBALCFGNAME_FIND_CONV_ALGO) == "TRUE") {
					find_fastest_cudnn_algo(nd);
				}
			}
		}
		else {
			CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());

			for (int nd = 0; nd < n_devices; nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					CHECK_EQ(inputs_[0]->get(nd)->channels(), channels_);

					outputs_[0]->get(nd)->reshape_like_4dext(
						*inputs_[0]->get(nd),
						num_outputs_,
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

	void conv_layer::resize_cudnn(int device_index) {
		int num = inputs_[0]->get(device_index)->num();
		int width = inputs_[0]->get(device_index)->width();
		int height = inputs_[0]->get(device_index)->height();
		int out_width = outputs_[0]->get(device_index)->width();
		int out_height = outputs_[0]->get(device_index)->height();

		int workspace_limit = out_width * out_height * ksize_w_ * ksize_h_ * channels_;
		workspace_limit *= layer_param_.conv_param().block_calc_size() == 0 ? 1 
			: layer_param_.conv_param().block_calc(0);
		workspace_->get(device_index)->reshape_4d(1, workspace_limit, 1, 1);

		cudnnHandle_t handle = context_->get_device(inputs_[0]->get(device_index)->device_id())->cudnn_handle();

		ENTER_DEVICE_CONTEXT(inputs_[0]->get(device_index)->device_id())

			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				conv_contexts_[device_index].bottom_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				num,
				channels_,
				height,
				width
			));

			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				conv_contexts_[device_index].top_desc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				num,
				num_outputs_,
				out_height,
				out_width
				));

			CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
				conv_contexts_[device_index].conv_desc,
				pad_h_,
				pad_w_,
				stride_h_,
				stride_w_,
				1,
				1,
				CUDNN_CROSS_CORRELATION
				));

			CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
				handle,
				conv_contexts_[device_index].bottom_desc,
				conv_contexts_[device_index].filter_desc,
				conv_contexts_[device_index].conv_desc,
				conv_contexts_[device_index].top_desc,
				CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
				workspace_limit * sizeof(data_type),
				&conv_contexts_[device_index].fwd_algo
				));

			if (context_->get_phase() == caffepro_context::TRAIN) {
				CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
					handle,
					conv_contexts_[device_index].bottom_desc,
					conv_contexts_[device_index].top_desc,
					conv_contexts_[device_index].conv_desc,
					conv_contexts_[device_index].filter_desc,
					CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
					workspace_limit * sizeof(data_type),
					&conv_contexts_[device_index].bwd_filter_algo
					));

				CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
					handle,
					conv_contexts_[device_index].filter_desc,
					conv_contexts_[device_index].top_desc,
					conv_contexts_[device_index].conv_desc,
					conv_contexts_[device_index].bottom_desc,
					CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
					workspace_limit * sizeof(data_type),
					&conv_contexts_[device_index].bwd_data_algo
					));
			}

			if (bias_term_) {
				CUDNN_CHECK(cudnnSetTensor4dDescriptor(
					conv_contexts_[device_index].bias_desc,
					CUDNN_TENSOR_NCHW,
					CUDNN_DATA_FLOAT,
					1,
					num_outputs_,
					1,
					1
					));
			}

		EXIT_DEVICE_CONTEXT;
	}

	void conv_layer::find_fastest_cudnn_algo(int device_index) {
		cudnnHandle_t handle = context_->get_device(inputs_[0]->get(device_index)->device_id())->cudnn_handle();
		const int max_num_algo = 20;
		const int max_workspace_size = 512 * 1024 * 1024;

		ENTER_DEVICE_CONTEXT(inputs_[0]->get(device_index)->device_id())

			int num_fwd_algo;
			cudnnConvolutionFwdAlgoPerf_t fwd_algo_perfs[max_num_algo];

			CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
				handle,
				conv_contexts_[device_index].bottom_desc,
				conv_contexts_[device_index].filter_desc,
				conv_contexts_[device_index].conv_desc,
				conv_contexts_[device_index].top_desc,
				max_num_algo,
				&num_fwd_algo,
				fwd_algo_perfs
				));

			if (num_fwd_algo > 0) {
				size_t workspace_size;
				CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
					handle, 
					conv_contexts_[device_index].bottom_desc,
					conv_contexts_[device_index].filter_desc,
					conv_contexts_[device_index].conv_desc,
					conv_contexts_[device_index].top_desc,
					fwd_algo_perfs[0].algo,
					&workspace_size
					));

				if (workspace_size <= workspace_->get(device_index)->count() * sizeof(data_type)) {
					conv_contexts_[device_index].fwd_algo = fwd_algo_perfs[0].algo;
				}
				else if (workspace_size <= max_workspace_size) {
					workspace_->get(device_index)->reshape_4d(1, (int)workspace_size / sizeof(data_type) + 1, 1, 1);
					conv_contexts_[device_index].fwd_algo = fwd_algo_perfs[0].algo;
				}
			}

			if (context_->get_phase() == caffepro_context::TRAIN) {
				int num_bwd_algo;
				cudnnConvolutionBwdDataAlgoPerf_t bwd_algo_prefs[max_num_algo];

				CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
					handle,
					conv_contexts_[device_index].filter_desc,
					conv_contexts_[device_index].top_desc,
					conv_contexts_[device_index].conv_desc,
					conv_contexts_[device_index].bottom_desc,
					max_num_algo,
					&num_bwd_algo,
					bwd_algo_prefs
					));

				if (num_bwd_algo > 0) {
					size_t workspace_size;
					CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
						handle,
						conv_contexts_[device_index].filter_desc,
						conv_contexts_[device_index].top_desc,
						conv_contexts_[device_index].conv_desc,
						conv_contexts_[device_index].bottom_desc,
						bwd_algo_prefs[0].algo,
						&workspace_size
						));

					if (workspace_size <= workspace_->get(device_index)->count() * sizeof(data_type)) {
						conv_contexts_[device_index].bwd_data_algo = bwd_algo_prefs[0].algo;
					}
					else if (workspace_size <= max_workspace_size) {
						workspace_->get(device_index)->reshape_4d(1, (int)workspace_size / sizeof(data_type) + 1, 1, 1);
						conv_contexts_[device_index].bwd_data_algo = bwd_algo_prefs[0].algo;
					}
				}

				int num_bwd_filter_algo;
				cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_prefs[max_num_algo];

				CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
					handle,
					conv_contexts_[device_index].bottom_desc,
					conv_contexts_[device_index].top_desc,
					conv_contexts_[device_index].conv_desc,
					conv_contexts_[device_index].filter_desc,
					max_num_algo,
					&num_bwd_filter_algo,
					bwd_filter_algo_prefs
					));

				if (num_bwd_filter_algo > 0) {
					size_t workspace_size;
					CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
						handle,
						conv_contexts_[device_index].bottom_desc,
						conv_contexts_[device_index].top_desc,
						conv_contexts_[device_index].conv_desc,
						conv_contexts_[device_index].filter_desc,
						bwd_filter_algo_prefs[0].algo,
						&workspace_size
						));

					if (workspace_size <= workspace_->get(device_index)->count() * sizeof(data_type)) {
						conv_contexts_[device_index].bwd_filter_algo = bwd_filter_algo_prefs[0].algo;
					}
					else if (workspace_size <= max_workspace_size) {
						workspace_->get(device_index)->reshape_4d(1, (int)workspace_size / sizeof(data_type) + 1, 1, 1);
						conv_contexts_[device_index].bwd_filter_algo = bwd_filter_algo_prefs[0].algo;
					}
				}
			}

		EXIT_DEVICE_CONTEXT;
	}

	void conv_layer::release_all() {
		for (auto pDes = conv_contexts_.begin(); pDes != conv_contexts_.end(); ++pDes) {
			if (pDes->bottom_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->bottom_desc));
			}
			if (pDes->top_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->top_desc));
			}
			if (pDes->filter_desc) {
				CUDNN_CHECK(cudnnDestroyFilterDescriptor(pDes->filter_desc));
			}
			if (pDes->conv_desc) {
				CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(pDes->conv_desc));
			}
			if (pDes->bias_desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(pDes->bias_desc));
			}

			memset(&*pDes, 0, sizeof(cudnn_conv_context));
		}
	}

	void conv_layer::on_forward(int device_index) {
		
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();
		const data_type alpha = 1, beta = 0;

		CUDNN_CHECK(cudnnConvolutionForward(
			handle,
			&alpha,
			conv_contexts_[device_index].bottom_desc,
			inputs_[0]->get(device_index)->gpu_data(),
			conv_contexts_[device_index].filter_desc,
			weights_[0]->get(device_index)->gpu_data(),
			conv_contexts_[device_index].conv_desc,
			conv_contexts_[device_index].fwd_algo,
			workspace_->get(device_index)->mutable_gpu_data(),
			workspace_->get(device_index)->count() * sizeof(data_type),
			&beta,
			conv_contexts_[device_index].top_desc,
			outputs_[0]->get(device_index)->mutable_gpu_data()
			));

		if (bias_term_) {
			CUDNN_CHECK(cudnnAddTensor(
				handle,
				&alpha,
				conv_contexts_[device_index].bias_desc,
				weights_[1]->get(device_index)->gpu_data(),
				&alpha,	// not beta
				conv_contexts_[device_index].top_desc,
				outputs_[0]->get(device_index)->mutable_gpu_data()
			));
		}
	}

	void conv_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type alpha = 1;
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		const data_type beta_weight = get_beta(clear_weights_diff, 0);
		const data_type beta_bias = get_beta(clear_weights_diff, 1);
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();

		if (bias_term_ && should_bp(bp_weights, 1)) {
			CUDNN_CHECK(cudnnConvolutionBackwardBias(
				handle,
				&alpha,
				conv_contexts_[device_index].top_desc,
				outputs_[0]->get(device_index)->gpu_diff(),
				&beta_bias,
				conv_contexts_[device_index].bias_desc,
				weights_[1]->get(device_index)->mutable_gpu_diff()
				));
		}

		if (should_bp(bp_weights, 0)) {
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(
				handle,
				&alpha,
				conv_contexts_[device_index].bottom_desc,
				inputs_[0]->get(device_index)->gpu_data(),
				conv_contexts_[device_index].top_desc,
				outputs_[0]->get(device_index)->gpu_diff(),
				conv_contexts_[device_index].conv_desc,
				conv_contexts_[device_index].bwd_filter_algo,
				workspace_->get(device_index)->mutable_gpu_data(),
				workspace_->get(device_index)->count() * sizeof(data_type),
				&beta_weight,
				conv_contexts_[device_index].filter_desc,
				weights_[0]->get(device_index)->mutable_gpu_diff()
				));
		}

		if (should_bp(bp_acts, 0)) {
			CUDNN_CHECK(cudnnConvolutionBackwardData(
				handle,
				&alpha,
				conv_contexts_[device_index].filter_desc,
				weights_[0]->get(device_index)->gpu_data(),
				conv_contexts_[device_index].top_desc,
				outputs_[0]->get(device_index)->gpu_diff(),
				conv_contexts_[device_index].conv_desc,
				conv_contexts_[device_index].bwd_data_algo,
				workspace_->get(device_index)->mutable_gpu_data(),
				workspace_->get(device_index)->count() * sizeof(data_type),
				&beta_acts,
				conv_contexts_[device_index].bottom_desc,
				inputs_[0]->get(device_index)->mutable_gpu_diff()
				));
		}

	}
}