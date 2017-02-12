
#include <caffepro/layers/runavg_batch_norm_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

#include <caffepro/context/common_names.h>
#include <caffepro/solver/sgd_solver.h>

namespace caffepro {
	runavg_batch_norm_layer::runavg_batch_norm_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	runavg_batch_norm_layer::~runavg_batch_norm_layer() {
		release_all();
	}

	void runavg_batch_norm_layer::release_all() {
		for (auto desc : feature_desc_) {
			if (desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
			}
		}

		for (auto desc : mean_var_desc_) {
			if (desc) {
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
			}
		}

		feature_desc_.clear();
		mean_var_desc_.clear();
	}

	void runavg_batch_norm_layer::init() {
		check_input();
		eps_ = CUDNN_BN_MIN_EPSILON;
		record_iters_ = 0;

		CHECK(layer_param_.batch_norm_param().has_sliding_window_eval_coeff())
			<< "sliding_window_eval_coeff mush be provided";
		running_factor_ = 1.f - layer_param_.batch_norm_param().sliding_window_eval_coeff();
		CHECK_GE(running_factor_, 0.f);
		CHECK_LE(running_factor_, 1.f);

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		channels_ = inputs_[0]->dim_at(2);

		// init blobs
		boost::shared_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, channels_, 1, 1));

		weights_.resize(4);
		for (int i = 0; i < 4; i++) {
			weights_[i].reset(new node_blob());
			weights_[i]->add_like(context_, *weight_template, *inputs_[0]);
		}

		// init scale (weights_[0])
		boost::shared_ptr<filler> scale_filler(filler::create(context_, layer_param_.batch_norm_param().scale_filler()));
		scale_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		// init shift (weights_[1])
		boost::shared_ptr<filler> shift_filler(filler::create(context_, layer_param_.batch_norm_param().shift_filler()));
		shift_filler->fill(*weights_[1]->get(0));
		weights_[1]->broadcast_data_via_gpu(0);

		// clear weights_[2] and weights_[3] (running avg mean and var)
		weights_[2]->get(0)->fill_data(0.f); // bn mean
		weights_[2]->get(0)->fill_diff(0.f);
		weights_[2]->broadcast_data_via_gpu(0);
		weights_[2]->broadcast_diff_via_gpu(0);

		weights_[3]->get(0)->fill_data(1.f); // bn var
		weights_[3]->get(0)->fill_diff(0.f);
		weights_[3]->broadcast_data_via_gpu(0); // bn std
		weights_[3]->broadcast_diff_via_gpu(0);

		EX_batch_.reset(new node_blob());
		EX_batch_->add_like(context_, *weight_template, *inputs_[0]);
		VarX_batch_.reset(new node_blob());
		VarX_batch_->add_like(context_, *weight_template, *inputs_[0]);
		internal_weights_.resize(2);
		internal_weights_[0] = EX_batch_;
		internal_weights_[1] = VarX_batch_;
		// do not set EX_batch and VarX_batch to NF_TEMP!

		EX_batch_->get(0)->fill_diff(0.f);
		VarX_batch_->get(0)->fill_diff(0.f);
		EX_batch_->broadcast_diff_via_gpu(0);
		VarX_batch_->broadcast_diff_via_gpu(0);

		init_cudnn();
	}

	void runavg_batch_norm_layer::init_cudnn() {
		int n_devices = (int)inputs_[0]->size();
		feature_desc_.resize(n_devices);
		mean_var_desc_.resize(n_devices);

		for (int nd = 0; nd < n_devices; nd++) {
			ENTER_DEVICE_CONTEXT(inputs_[0]->get(nd)->device_id())
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&feature_desc_[nd]));
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&mean_var_desc_[nd]));

				CUDNN_CHECK(cudnnSetTensor4dDescriptor(mean_var_desc_[nd], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
					1, channels_, 1, 1));
			EXIT_DEVICE_CONTEXT
		}
	}

	void runavg_batch_norm_layer::resize() {
		caffepro_layer::resize();

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		CHECK_EQ(channels_, inputs_[0]->dim_at(2));

		for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				auto &input = *inputs_[0]->get(nd);
				ENTER_DEVICE_CONTEXT(inputs_[0]->get(nd)->device_id())
					CUDNN_CHECK(cudnnSetTensor4dDescriptor(feature_desc_[nd], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						input.num(), input.channels(), input.height(), input.width()));
				EXIT_DEVICE_CONTEXT
			}
		}
	}

	void runavg_batch_norm_layer::on_forward(int device_index) {
		const data_type alpha = 1.f, beta = 0.f;
		if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD) {
			CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
				context_->get_current_device()->cudnn_handle(),
				CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				feature_desc_[device_index],
				inputs_[0]->get(device_index)->gpu_data(),
				feature_desc_[device_index],
				outputs_[0]->get(device_index)->mutable_gpu_data(),
				mean_var_desc_[device_index],
				weights_[0]->get(device_index)->gpu_data(),
				weights_[1]->get(device_index)->gpu_data(),
				running_factor_,
				weights_[2]->get(device_index)->mutable_gpu_data(),
				weights_[3]->get(device_index)->mutable_gpu_data(),
				eps_,
				EX_batch_->get(device_index)->mutable_gpu_data(),
				VarX_batch_->get(device_index)->mutable_gpu_data()
				));
		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD) {
			CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
				context_->get_current_device()->cudnn_handle(),
				CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				feature_desc_[device_index],
				inputs_[0]->get(device_index)->gpu_data(),
				feature_desc_[device_index],
				outputs_[0]->get(device_index)->mutable_gpu_data(),
				mean_var_desc_[device_index],
				weights_[0]->get(device_index)->gpu_data(),
				weights_[1]->get(device_index)->gpu_data(),
				1, // do not use running avg
				EX_batch_->get(device_index)->mutable_gpu_diff(), // cache mean and variance to EX and VarX diff
				VarX_batch_->get(device_index)->mutable_gpu_diff(),
				eps_,
				EX_batch_->get(device_index)->mutable_gpu_data(),
				VarX_batch_->get(device_index)->mutable_gpu_data()
				));
		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_USE_RECORD_NORM) {
			CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
				context_->get_current_device()->cudnn_handle(),
				CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				feature_desc_[device_index],
				inputs_[0]->get(device_index)->gpu_data(),
				feature_desc_[device_index],
				outputs_[0]->get(device_index)->mutable_gpu_data(),
				mean_var_desc_[device_index],
				weights_[0]->get(device_index)->gpu_data(),
				weights_[1]->get(device_index)->gpu_data(),
				weights_[2]->get(device_index)->gpu_data(),
				weights_[3]->get(device_index)->gpu_data(),
				eps_
				));
		}
		else {
			LOG(FATAL) << "Not supported for RECORD mode";
		}
	}

	void runavg_batch_norm_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0) || should_bp(bp_weights, 0) || should_bp(bp_weights, 1)) {
			CHECK(layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD
				|| layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD);
			
			const data_type alpha = 1.f;
			const data_type beta_acts = should_bp(bp_acts, 0) ? get_beta(clear_acts_diff, 0) : 0;
			const data_type beta_scale = should_bp(bp_weights, 0) ? get_beta(clear_weights_diff, 0) : 0;
			const data_type beta_shift = should_bp(bp_weights, 1) ? get_beta(clear_weights_diff, 1) : 0;
			CHECK_EQ(beta_scale, beta_shift);

			CUDNN_CHECK(cudnnBatchNormalizationBackward(
				context_->get_current_device()->cudnn_handle(),
				CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta_acts,
				&alpha,
				&beta_scale, // beta scale == beta shift
				feature_desc_[device_index],
				inputs_[0]->get(device_index)->gpu_data(),
				feature_desc_[device_index],
				outputs_[0]->get(device_index)->gpu_diff(),
				feature_desc_[device_index],
				inputs_[0]->get(device_index)->mutable_gpu_diff(),
				mean_var_desc_[device_index],
				weights_[0]->get(device_index)->gpu_data(),
				weights_[0]->get(device_index)->mutable_gpu_diff(),
				weights_[1]->get(device_index)->mutable_gpu_diff(),
				eps_,
				EX_batch_->get(device_index)->gpu_data(),
				VarX_batch_->get(device_index)->gpu_data()
				));
		}
	}

	void runavg_batch_norm_layer::on_after_forward() {
		if (layer_param_.batch_norm_param().record_option() != BatchNormalizationParameter_RecordOption_RECORD) {
			return;
		}

		CHECK_EQ(context_->get_phase(), caffepro_context::TRAIN);
		context_->sync_all_devices();

		int count = weights_[2]->get(0)->count();
		int n_gpus = (int)EX_batch_->size();

		ENTER_DEVICE_CONTEXT(inputs_[0]->get(0)->device_id())
			if (record_iters_ == 0) {
				// clear record EX_ and EX2_
				CUDA_CHECK(cudaMemsetAsync(weights_[2]->get(0)->mutable_gpu_data(), 0, count * sizeof(data_type)));
				CUDA_CHECK(cudaMemsetAsync(weights_[3]->get(0)->mutable_gpu_data(), 0, count * sizeof(data_type)));
			}

			record_iters_++;

			int cur_iters = record_iters_;
			data_type previous_scale = (data_type)(cur_iters - 1) / (data_type)cur_iters;
			data_type current_scale_factor = (data_type)1. / (data_type)n_gpus / (data_type)cur_iters;

			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

			cublas.axpby(
				count,
				current_scale_factor,
				EX_batch_->get(0)->gpu_diff(), // EX cached in EX_batch_diff 
				previous_scale,
				weights_[2]->get(0)->mutable_gpu_data()
				);

			cublas.axpby(
				count,
				current_scale_factor,
				VarX_batch_->get(0)->gpu_diff(), // VarX cached in VarX_batch_diff
				previous_scale,
				weights_[3]->get(0)->mutable_gpu_data()
				);

			for (int nd = 1; nd < n_gpus; nd++) {
				// copy to gpu 0
				CHECK_EQ(EX_batch_->get(nd)->count(), count);
				CHECK_EQ(VarX_batch_->get(nd)->count(), count);

				CUDA_CHECK(cudaMemcpyAsync(
					EX_batch_->get(0)->mutable_gpu_diff(),
					EX_batch_->get(nd)->gpu_diff(),
					count * sizeof(data_type),
					cudaMemcpyDeviceToDevice
					));

				CUDA_CHECK(cudaMemcpyAsync(
					VarX_batch_->get(0)->mutable_gpu_diff(),
					VarX_batch_->get(nd)->gpu_diff(),
					count * sizeof(data_type),
					cudaMemcpyDeviceToDevice
					)); // here, EX2 are store in VarX_batch_diff

				// merge EX and EX2
				cublas.axpy(count, current_scale_factor, EX_batch_->get(0)->gpu_diff(),
					weights_[2]->get(0)->mutable_gpu_data());

				cublas.axpy(count, current_scale_factor, VarX_batch_->get(0)->gpu_diff(),
					weights_[3]->get(0)->mutable_gpu_data());
			}
		EXIT_DEVICE_CONTEXT
	}
}