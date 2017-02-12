
#include <caffepro/layers/batch_norm_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/math/cublas_debug.h>
#include <caffepro/utils/utils.h>

#include <caffepro/context/common_names.h>
#include <caffepro/solver/sgd_solver.h>

namespace caffepro {
	batch_norm_layer::batch_norm_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	batch_norm_layer::~batch_norm_layer() {
		release_all();
	}

	void batch_norm_layer::init() {
		check_input();

		eps_ = 1e-9f;
		record_iters_ = 0;
		batch_size_ = inputs_[0]->sum_num();
		keep_mean_ = layer_param_.batch_norm_param().keep_mean();

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

		// clear weights_[2] and weights_[3]
		weights_[2]->get(0)->fill_data(0.f); // bn mean
		weights_[2]->broadcast_data_via_gpu(0);

		//weights_[3]->get(0)->fill_data(0.f); // bn E[(x-E[x])^2]
		weights_[3]->get(0)->fill_data(1.f); // bn E[(x-E[x])^2]
		weights_[3]->broadcast_data_via_gpu(0);
		//LOG(INFO) << "Warning: BN: weights_[3]->get(0)->fill_data(1.f)";

		// setup internal weights
		sum_multiplier_.reset(new node_blob());
		sum_multiplier_num_.reset(new node_blob());

		internal_weights_.resize(4);
		EX_.reset(new node_blob());
		EX2_.reset(new node_blob());
		EX_->set_attr(node_blob::NF_TEMP);
		EX2_->set_attr(node_blob::NF_TEMP);
		internal_weights_[0] = EX_;
		internal_weights_[1] = EX2_;

		EX_batch_.reset(new node_blob());
		EX_batch_->add_like(context_, *weight_template, *inputs_[0]);
		VarX_batch_.reset(new node_blob());
		VarX_batch_->add_like(context_, *weight_template, *inputs_[0]);
		internal_weights_[2] = EX_batch_;
		internal_weights_[3] = VarX_batch_;
		// do not set EX_batch and VarX_batch to NF_TEMP!
	}

	void batch_norm_layer::resize() {
		caffepro_layer::resize();

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		CHECK_EQ(channels_, inputs_[0]->dim_at(2));

		if (EX_->size() == 0) { // first run
			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				EX_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, inputs_[0]->get(nd)->num(), channels_, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				EX2_->add(boost::shared_ptr<device_blob>(
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
						EX2_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), channels_, 1, 1);
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

	__global__ static void batch_norm_forward(const int n, const int spatial_size, const int channels,
		const data_type *bottom_data,
		const data_type *EX_batch_data, const data_type *VarX_batch_data,
		const data_type *scale_data, const data_type *shift_data,
		data_type *top_data, const bool keep_mean) {

		CUDA_KERNEL_LOOP(index, n) {
			const int c = (index / spatial_size) % channels;

			data_type v_input_norm = (bottom_data[index] - EX_batch_data[c]) / VarX_batch_data[c];
			data_type v = v_input_norm  * scale_data[c] + shift_data[c];
			if (keep_mean) {
				v += EX_batch_data[c];
			}
			top_data[index] = v;
		}
	}

	void batch_norm_layer::on_forward(int device_index) {
		int num = inputs_[0]->get(device_index)->num();
		int channels = inputs_[0]->get(device_index)->channels();
		int num_of_vecs = num * channels;
		int spatial_size = inputs_[0]->get(device_index)->height() * inputs_[0]->get(device_index)->width();
		data_type *EX_data = EX_->get(device_index)->mutable_gpu_data();
		data_type *EX2_data = EX2_->get(device_index)->mutable_gpu_data();
		data_type *EX_batch_data = EX_batch_->get(device_index)->mutable_gpu_data();
		data_type *VarX_batch_data = VarX_batch_->get(device_index)->mutable_gpu_data();

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD
			|| layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD) {
			// In this case, EX_batch and VarX_batch should be calculated from data on current GPU

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

			// calc VarX_batch
			cublas.mul(channels, EX_batch_data, EX_batch_data, VarX_batch_data); // (EX)^2

			cublas.mul(
				inputs_[0]->get(device_index)->count(), 
				inputs_[0]->get(device_index)->gpu_data(), 
				inputs_[0]->get(device_index)->gpu_data(), 
				outputs_[0]->get(device_index)->mutable_gpu_data()
				); // X^2

			cublas.gemv(CblasNoTrans,
				num_of_vecs,
				spatial_size,
				(data_type)1.f / (data_type)spatial_size,
				outputs_[0]->get(device_index)->gpu_data(),
				sum_multiplier_->get(device_index)->gpu_data(),
				(data_type)0.f,
				EX2_data
				); // calc EX2

			cublas.gemv(
				CblasTrans,
				num,
				channels,
				(data_type)1.f / (data_type)num,
				EX2_data,
				sum_multiplier_num_->get(device_index)->gpu_data(),
				(data_type)-1.f,
				VarX_batch_data
				); // calc VarX_batch (VarX = E(X^2) - (EX)^2)

			cublas.max_scalar(channels, (data_type)0.f, VarX_batch_data); // preventing numerical problems
			cublas.add_scalar(channels, eps_, VarX_batch_data); // VarX_batch += eps

			if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD) {
				data_type sliding_window_eval_coeff = layer_param_.batch_norm_param().sliding_window_eval_coeff();
				if (sliding_window_eval_coeff >= 0.f && sliding_window_eval_coeff <= 1.f) {
					// sliding window mean and variance statistic
					
					// force to 0 if too small
					if (sliding_window_eval_coeff < 1e-7) {
						sliding_window_eval_coeff = 0; //set zero
						weights_[2]->get(device_index)->fill_data(0.f);
						weights_[3]->get(device_index)->fill_data(1.f);
						//LOG(ERROR) << "Warning: BN: weights_[2]->get(device_index)->fill_data(0.f); weights_[3]->get(0)->fill_data(1.f);";
					}

					// set sliding_window_eval_coeff = 0 based on lr			
					sgd_solver* sgd_solver_ = dynamic_cast<sgd_solver*>(this->context()->get_shared_object(namespace_, SHAREDOBJNAME_SGDSOLVER));
					if (sgd_solver_ != nullptr) {
						data_type lr = sgd_solver_->get_learning_rate();
						if (lr >= 0.01f) sliding_window_eval_coeff = 0; // do not perform running aveage on the first two stages
						
						// for debug only
						int iter = sgd_solver_->get_iter();
						if ((iter % 2500 == 0 || iter % 2500 == 1 || iter % 2500 == 20) && (device_index == 0) && (this->layer_param_.name().find("conv1") != std::string::npos))
							LOG(INFO) << "Warning: iter: " << iter << ", lr: " << lr << ", eval_coeff: " << sliding_window_eval_coeff << ", layer: " << this->layer_param_.name();
					}
					
					// compute
					cublas.axpby(channels, 1 - sliding_window_eval_coeff, EX_batch_data, sliding_window_eval_coeff, weights_[2]->get(device_index)->mutable_gpu_data());
					cublas.axpby(channels, 1 - sliding_window_eval_coeff, VarX_batch_data, sliding_window_eval_coeff, weights_[3]->get(device_index)->mutable_gpu_data());
				}
			}

			cublas.sqrt(channels, VarX_batch_data, VarX_batch_data); // VarX_batch = sqrt(VarX_batch)

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

				cublas.gemv(
					CblasTrans,
					num,
					channels,
					(data_type)1.f, // NOT (1 / num) !!!
					EX2_data,
					sum_multiplier_num_->get(device_index)->gpu_data(),
					(data_type)0.f,
					VarX_batch_->get(device_index)->mutable_gpu_diff()
					);
			}
		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_USE_RECORD_NORM) {
			// In this case, EX_batch and VarX_batch are fixed

			cublas.copy(channels, weights_[2]->get(device_index)->gpu_data(), EX_batch_data); // load EX_batch

			cublas.mul(channels, EX_batch_data, EX_batch_data, VarX_batch_data); // (EX)^2
			cublas.axpby(channels, (data_type)1.f, weights_[3]->get(device_index)->gpu_data(), (data_type)-1.f, VarX_batch_data); // VarX_batch = E(X^2) - (EX)^2
			cublas.max_scalar(channels, (data_type)0.f, VarX_batch_data); // preventing numerical problems
			cublas.add_scalar(channels, eps_, VarX_batch_data); // VarX_batch += eps
			cublas.sqrt(channels, VarX_batch_data, VarX_batch_data); // VarX_batch = sqrt(VarX_batch)
		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_CALC) {
			// for validation purpose only
			// in this case, we needn't calculate EX_batch and VarX_batch because they have been processed in the last training batch

			//LOG(INFO) << layer_param_.batch_norm_param().record_option() << std::endl;
			CHECK_EQ(context_->get_phase(), caffepro_context::TEST);

			data_type sliding_window_eval_coeff = layer_param_.batch_norm_param().sliding_window_eval_coeff();
			if (sliding_window_eval_coeff >= 0.f && sliding_window_eval_coeff <= 1.f) {
				// if sliding window evaluation enabled, use these data
				cublas.copy(channels, weights_[2]->get(device_index)->gpu_data(), EX_batch_data);
				//cublas.copy(channels, weights_[3]->get(device_index)->gpu_data(), VarX_batch_data);
				cublas.sqrt(channels, weights_[3]->get(device_index)->gpu_data(), VarX_batch_data);
			}
		}
		else {
			NOT_IMPLEMENTED;
		}

		// do forward
		int count = inputs_[0]->get(device_index)->count();
		KERNEL_CALL(batch_norm_forward, count)(
			count, spatial_size, channels,
			inputs_[0]->get(device_index)->gpu_data(), EX_batch_data, VarX_batch_data,
			weights_[0]->get(device_index)->gpu_data(), weights_[1]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data(),
			keep_mean_
			);

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void get_batch_norm(const int n, const int spatial_size, const int channels,
		const data_type *input_data, const data_type *EX_batch_data, const data_type *VarX_batch_data,
		data_type *output_data) {
		CUDA_KERNEL_LOOP(index, n) {
			const int c = (index / spatial_size) % channels;

			data_type v_input_norm = (input_data[index] - EX_batch_data[c]) / VarX_batch_data[c];
			output_data[index] = v_input_norm;
		}
	}

	__global__ static void batch_norm_backward(const int n, const int spatial_size, const int channels,
		const data_type *top_diff, const data_type *batch_sum_top_diff, const data_type *normed_bottom_data,
		const data_type *batch_sum_top_diff_norm_bottom_data,
		const data_type *VarX_batch_data, const data_type *scale_data,
		data_type *bottom_diff, const bool keep_mean) {
		// WARNING: bottom_diff and normed_bottom_data shares the same space

		CUDA_KERNEL_LOOP(index, n) {
			const int num = n / spatial_size / channels;
			const int n_sum = num * spatial_size;

			const int c = (index / spatial_size) % channels;

			data_type v_var = VarX_batch_data[c];
			data_type v_sum_top_diff = batch_sum_top_diff[c];
			data_type v_sum_diff_norm_bottom = batch_sum_top_diff_norm_bottom_data[c];

			data_type v_top_diff = top_diff[index];
			data_type v_norm_bottom = normed_bottom_data[index];

			data_type v_scale = scale_data[c];

			data_type v = (v_top_diff / v_var
				- v_sum_top_diff / v_var / (data_type)n_sum
				- v_norm_bottom * v_sum_diff_norm_bottom / v_var / (data_type)n_sum
				) * v_scale;

			if (keep_mean) {
				v += v_sum_top_diff / n_sum;
			}

			bottom_diff[index] = v;
		}
	}

	__global__ static void batch_norm_backward_use_record_norm(const int n, const int spatial_size, const int channels,
		const data_type *top_diff, const data_type *VarX_data, const data_type *scale_data,
		data_type *bottom_diff) { // keep_mean not needed here
		// WARNING: bottom_diff and normed_bottom_data shares the same space

		CUDA_KERNEL_LOOP(index, n) {
			const int c = (index / spatial_size) % channels;

			data_type v_var = VarX_data[c];
			data_type v_scale = scale_data[c];
			data_type v_top_diff = top_diff[index];

			bottom_diff[index] = v_top_diff / v_var * v_scale;
		}
	}

	void batch_norm_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		data_type beta_scale = get_beta(clear_weights_diff, 0);
		data_type beta_shift = get_beta(clear_weights_diff, 1);

		if (should_bp(bp_acts, 0)) {
			CHECK_EQ(beta_acts, 0.f) << "Currently you can not fork the input of a batch norm layer";
		}

		// prevent bug
		if (!should_bp(bp_weights, 0)) {
			beta_scale = 0.f;
		}

		if (!should_bp(bp_weights, 1)) {
			beta_shift = 0.f;
		}

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		int num = inputs_[0]->get(device_index)->num();
		int channels = inputs_[0]->get(device_index)->channels();
		int num_of_vecs = num * channels;
		int spatial_size = inputs_[0]->get(device_index)->height() * inputs_[0]->get(device_index)->width();
		int count = inputs_[0]->get(device_index)->count();
		data_type *EX_batch_data = EX_batch_->get(device_index)->mutable_gpu_data();
		data_type *VarX_batch_data = VarX_batch_->get(device_index)->mutable_gpu_data();

		data_type *EX_diff = EX_->get(device_index)->mutable_gpu_diff();
		data_type *EX2_diff = EX2_->get(device_index)->mutable_gpu_diff();

		// step 1, calc sum_spatial(top_diff)) and sum_spatial(top_diff .* norm(input))) 
		// bp scale and shift at the same time

		KERNEL_CALL(get_batch_norm, count)(
			count, spatial_size, channels, inputs_[0]->get(device_index)->gpu_data(), EX_batch_data, VarX_batch_data,
			inputs_[0]->get(device_index)->mutable_gpu_diff() // cache norm(input) into bottom_diff
			);

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
			beta_shift,	// should be 0 unless for USE_RECORD_NORM
			weights_[1]->get(device_index)->mutable_gpu_diff()
			); // bp shift, i.e. shift_diff = sum_num(sum_spatial(top_diff))

		cublas.mul(outputs_[0]->get(device_index)->count(), outputs_[0]->get(device_index)->gpu_diff(), inputs_[0]->get(device_index)->gpu_diff(),
			outputs_[0]->get(device_index)->mutable_gpu_data()); // cache (top_diff .* norm(input)) into top_data

		cublas.gemv(
			CblasNoTrans,
			num_of_vecs,
			spatial_size,
			(data_type)1.f,
			outputs_[0]->get(device_index)->gpu_data(),
			sum_multiplier_->get(device_index)->gpu_data(),
			(data_type)0.f,
			EX2_diff
			); // cache sum_spatial(top_diff .* norm(input)) into EX2_diff

		cublas.gemv(
			CblasTrans,
			num,
			channels,
			(data_type)1.f,
			EX2_diff,
			sum_multiplier_num_->get(device_index)->gpu_data(),
			beta_scale,	// should be 0 unless for USE_RECORD_NORM
			weights_[0]->get(device_index)->mutable_gpu_diff()
			); // bp scale, i.e. scale_diff = sum_num(sum_spatial(top_diff .* norm(input)))

		if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_NOT_RECORD
			|| layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_RECORD) {
			// step 2: bp acts

			if (should_bp(bp_weights, 0)) {
				CHECK_EQ(beta_scale, 0) << "Cannot share the scale data of BN layer";
			}
			if (should_bp(bp_weights, 1)) {
				CHECK_EQ(beta_shift, 0) << "Cannot share the shift data of BN layer";
			}

			KERNEL_CALL(batch_norm_backward, count)(
				count,
				spatial_size,
				channels,
				outputs_[0]->get(device_index)->gpu_diff(),
				weights_[1]->get(device_index)->gpu_diff(), // stores sum_num(sum_spatial(top_diff))
				inputs_[0]->get(device_index)->gpu_diff(), // stores norm(input)
				weights_[0]->get(device_index)->gpu_diff(), // stores sum_norm(sum_spatial(top_diff .* norm(input))) 
				VarX_batch_data,
				weights_[0]->get(device_index)->gpu_data(), // stores scale data
				inputs_[0]->get(device_index)->mutable_gpu_diff(), // OVERRIDE norm(input) here is OK
				keep_mean_
				);

		}
		else if (layer_param_.batch_norm_param().record_option() == BatchNormalizationParameter_RecordOption_USE_RECORD_NORM) {
			// step 2: bp acts

			KERNEL_CALL(batch_norm_backward_use_record_norm, count)(
				count,
				spatial_size,
				channels,
				outputs_[0]->get(device_index)->gpu_diff(),
				VarX_batch_data,	// here, VarX_batch was derived from blobs_[3] in forward pass
				weights_[0]->get(device_index)->gpu_data(),
				inputs_[0]->get(device_index)->mutable_gpu_diff()
				); // keep_mean not needed here
		}
		else {
			NOT_IMPLEMENTED;
		}

		CUDA_POST_KERNEL_CHECK;
	}

	void batch_norm_layer::on_after_forward() {
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
				weights_[2]->get(0)->mutable_gpu_data()
				);

			cublas.axpby(
				count,
				current_scale_factor,
				VarX_batch_->get(0)->gpu_diff(), // EX2 cached in VarX_batch_diff
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