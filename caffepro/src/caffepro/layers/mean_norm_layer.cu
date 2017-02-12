
#include <caffepro/layers/mean_norm_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	mean_norm_layer::mean_norm_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	mean_norm_layer::~mean_norm_layer() {
		release_all();
	}

	void mean_norm_layer::init() {
		check_input();

		record_iters_ = 0;
		batch_size_ = inputs_[0]->sum_num();

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		channels_ = inputs_[0]->dim_at(2);

		// init blobs
		boost::shared_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, channels_, 1, 1));

		weights_.resize(2);
		for (int i = 0; i < 2; i++) {
			weights_[i].reset(new node_blob());
			weights_[i]->add_like(context_, *weight_template, *inputs_[0]);
		}

		// init shift (weights_[0])
		boost::shared_ptr<filler> shift_filler(filler::create(context_, layer_param_.batch_norm_param().shift_filler()));
		shift_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		// clear weights_[1]
		weights_[1]->get(0)->fill_data(0.f); // bn mean
		weights_[1]->broadcast_data_via_gpu(0);

		// setup internal weights
		sum_multiplier_.reset(new node_blob());
		sum_multiplier_num_.reset(new node_blob());

		internal_weights_.resize(2);
		EX_.reset(new node_blob());
		EX_->set_attr(node_blob::NF_TEMP);
		internal_weights_[0] = EX_;

		EX_batch_.reset(new node_blob());
		EX_batch_->add_like(context_, *weight_template, *inputs_[0]);
		internal_weights_[1] = EX_batch_;
		// do not set EX_batch NF_TEMP!
	}

	void mean_norm_layer::resize() {
		caffepro_layer::resize();

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		CHECK_EQ(channels_, inputs_[0]->dim_at(2));

		if (EX_->size() == 0) { // first run
			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				EX_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, inputs_[0]->get(nd)->num(), channels_, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				int spatial_size = inputs_[0]->get(nd)->width() * inputs_[0]->get(nd)->height();

				sum_multiplier_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, 1, spatial_size, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				sum_multiplier_num_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, inputs_[0]->get(nd)->num(), 1, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				sum_multiplier_->get(nd)->fill_data(1.f);
				sum_multiplier_num_->get(nd)->fill_data(1.f);
			}
		}
		else {
			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					int num = inputs_[0]->get(nd)->num();

					if (num != EX_->get(nd)->num()) {
						EX_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), channels_, 1, 1);
						sum_multiplier_num_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), 1, 1, 1);
						sum_multiplier_num_->get(nd)->fill_data(1.f);
					}

					int spatial_size = inputs_[0]->get(nd)->width() * inputs_[0]->get(nd)->height();
					if (spatial_size != sum_multiplier_->get(nd)->count()) {
						sum_multiplier_->get(nd)->reshape_4d(1, spatial_size, 1, 1);
						sum_multiplier_->get(nd)->fill_data(1.f);
					}
				}
			}
		}
	}

	__global__ static void mean_norm_forward(const int n, const int spatial_size, const int channels,
		const data_type *bottom_data, const data_type *EX_batch_data, const data_type *shift_data,
		data_type *top_data) {

		CUDA_KERNEL_LOOP(index, n) {
			const int c = (index / spatial_size) % channels;

			data_type v_input_norm = bottom_data[index] - EX_batch_data[c];
			top_data[index] = v_input_norm + shift_data[c];
		}
	}

	void mean_norm_layer::on_forward(int device_index) {
		int num = inputs_[0]->get(device_index)->num();
		int channels = inputs_[0]->get(device_index)->channels();
		int num_of_vecs = num * channels;
		int spatial_size = inputs_[0]->get(device_index)->height() * inputs_[0]->get(device_index)->width();
		data_type *EX_data = EX_->get(device_index)->mutable_gpu_data();
		data_type *EX_batch_data = EX_batch_->get(device_index)->mutable_gpu_data();

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD
			|| layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD) {
			// In this case, EX_batch should be calculated from data on current GPU

			// calc EX
			cublas.gemv(
				CblasNoTrans,
				num_of_vecs,
				spatial_size,
				(data_type)1.f / (data_type)spatial_size,
				inputs_[0]->get(device_index)->gpu_data(),
				sum_multiplier_->get(device_index)->gpu_data(),
				(data_type)0.f,
				EX_data
				);

			// calc EX_batch
			cublas.gemv(
				CblasTrans,
				num,
				channels,
				(data_type)1.f / (data_type)num,
				EX_data,
				sum_multiplier_num_->get(device_index)->gpu_data(),
				(data_type)0.f,
				EX_batch_data
				);

			if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD) {
				// backup EX_batch and EX2_batch to diff (add directly, NO 1 / num)

				cublas.gemv(
					CblasTrans,
					num,
					channels,
					(data_type)1., // NOT (1 / num) !!! 
					EX_data,
					sum_multiplier_num_->get(device_index)->gpu_data(),
					(data_type)0.,
					EX_batch_->get(device_index)->mutable_gpu_diff()
					);
			}
		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_USE_RECORD_NORM) {
			// In this case, EX_batch and VarX_batch are fixed

			cublas.copy(channels, weights_[1]->get(device_index)->gpu_data(), EX_batch_data); // load EX_batch
		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_CALC) {
			// for validation purpose only
			// in this case, we needn't calculate EX_batch and VarX_batch because they have been processed in the last training batch

			CHECK_EQ(context_->get_phase(), caffepro_context::TEST);
		}
		else {
			NOT_IMPLEMENTED;
		}

		// do forward
		int count = inputs_[0]->get(device_index)->count();
		KERNEL_CALL(mean_norm_forward, count)(
			count, spatial_size, channels,
			inputs_[0]->get(device_index)->gpu_data(), EX_batch_data,
			weights_[0]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data()
			);

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void mean_norm_backward(const int n, const int spatial_size, const int channels,
		const data_type *top_diff, const data_type *batch_sum_top_diff, 
		data_type *bottom_diff, const data_type scale_targets) {

		CUDA_KERNEL_LOOP(index, n) {
			const int num = n / spatial_size / channels;
			const int n_sum = num * spatial_size;

			const int c = (index / spatial_size) % channels;
			data_type v_sum_top_diff = batch_sum_top_diff[c];

			data_type v_top_diff = top_diff[index];
			data_type v = v_top_diff - v_sum_top_diff / (data_type)n_sum;
			
			if (scale_targets == 0) {
				bottom_diff[index] = v;
			}
			else {
				bottom_diff[index] = bottom_diff[index] * scale_targets + v;
			}
		}
	}

	void mean_norm_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		const data_type beta_shift = get_beta(clear_weights_diff, 0);

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		int num = inputs_[0]->get(device_index)->num();
		int channels = inputs_[0]->get(device_index)->channels();
		int num_of_vecs = num * channels;
		int spatial_size = inputs_[0]->get(device_index)->height() * inputs_[0]->get(device_index)->width();
		int count = inputs_[0]->get(device_index)->count();
		data_type *EX_batch_data = EX_batch_->get(device_index)->mutable_gpu_data();
		data_type *EX_batch_diff = EX_batch_->get(device_index)->mutable_gpu_diff();
		data_type *EX_diff = EX_->get(device_index)->mutable_gpu_diff();

		cublas.gemv(
			CblasNoTrans,
			num_of_vecs,
			spatial_size,
			(data_type)1.f,
			outputs_[0]->get(device_index)->gpu_diff(),
			sum_multiplier_->get(device_index)->gpu_data(),
			(data_type)0.f,
			EX_diff
			); // cache sum_spatial(top_diff) into EX_diff

		cublas.gemv(
			CblasTrans,
			num,
			channels,
			(data_type)1.f,
			EX_diff,
			sum_multiplier_num_->get(device_index)->gpu_data(),
			0,
			EX_batch_diff
			); // sum_num(sum_spatial(top_diff))

		if (should_bp(bp_weights, 0)) {
			cublas.axpby(num, 1.f, EX_batch_diff, beta_shift, weights_[0]->get(device_index)->mutable_gpu_diff());
		}

		if (should_bp(bp_acts, 0)) {
			if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD
				|| layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD) {
				// step 2: bp acts
				KERNEL_CALL(mean_norm_backward, count)(
					count,
					spatial_size,
					channels,
					outputs_[0]->get(device_index)->gpu_diff(),
					EX_batch_diff, // sum_num(sum_spatial(top_diff)) 
					inputs_[0]->get(device_index)->mutable_gpu_diff(),
					beta_acts
					);

			}
			else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_USE_RECORD_NORM) {
				// step 2: bp acts
				cublas.axpby(count, 1.f, outputs_[0]->get(device_index)->gpu_diff(), beta_acts, inputs_[0]->get(device_index)->mutable_gpu_diff());
			}
			else {
				NOT_IMPLEMENTED;
			}
		}

		CUDA_POST_KERNEL_CHECK;
	}

	void mean_norm_layer::on_after_forward() {
		if (layer_param_.batch_norm_param().record_option() != BatchNormalizationParameter_RecordOption_RECORD) {
			return;
		}

		CHECK_EQ(context_->get_phase(), caffepro_context::TRAIN);
		context_->sync_all_devices();

		int count = weights_[1]->get(0)->count();
		int n_gpus = (int)EX_batch_->size();

		ENTER_DEVICE_CONTEXT(inputs_[0]->get(0)->device_id())
		if (record_iters_ == 0) {
			// clear record EX_
			CUDA_CHECK(cudaMemsetAsync(weights_[1]->get(0)->mutable_gpu_data(), 0, count * sizeof(data_type)));

			batch_size_ = inputs_[0]->sum_num();
		}

		record_iters_++;

		CHECK_EQ(inputs_[0]->sum_num(), batch_size_);
		int cur_iters = record_iters_;
		data_type previous_scale = (data_type)(cur_iters - 1) / (data_type)cur_iters;
		data_type current_scale_factor = (data_type)1. / (data_type)batch_size_ / (data_type)cur_iters;

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		cublas.axpby(
			count,
			current_scale_factor,
			EX_batch_->get(0)->gpu_diff(), // EX cached in EX_batch_diff 
			previous_scale,
			weights_[1]->get(0)->mutable_gpu_data()
			);

		for (int nd = 1; nd < n_gpus; nd++) {
			// copy to gpu 0
			CHECK_EQ(EX_batch_->get(nd)->count(), count);

			CUDA_CHECK(cudaMemcpyAsync(
				EX_batch_->get(0)->mutable_gpu_diff(),
				EX_batch_->get(nd)->gpu_diff(),
				count * sizeof(data_type),
				cudaMemcpyDeviceToDevice
				));

			// merge EX and EX2
			cublas.axpy(count, current_scale_factor, EX_batch_->get(0)->gpu_diff(),
				weights_[1]->get(0)->mutable_gpu_data());
		}
		EXIT_DEVICE_CONTEXT
	}
}