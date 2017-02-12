
#include <caffepro/layers/learnable_dropout_layer.h>
#include <caffepro/layers/sigmoid_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <boost/scoped_ptr.hpp>

namespace caffepro {
	learnable_dropout_layer::learnable_dropout_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_INNER_COUNT_ACROSS_DEVICES
			);
	}

	learnable_dropout_layer::~learnable_dropout_layer() {
		release_all();
	}

	void learnable_dropout_layer::init() {
		check_input();

		boost::scoped_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, 1, 1, 1));
		weights_.push_back(boost::shared_ptr<node_blob>(new node_blob()));
		weights_[0]->add_like(*weight_template, *inputs_[0]);

		weights_[0]->get(0)->fill_data(layer_param_.learnable_dropout_param().init_value());
		weights_[0]->broadcast_data_via_gpu(0);

		thres_.reset(new node_blob()); // thres_ is reserved ratio, not dropout ratio!
		sigmoid_inputs_.push_back(weights_[0]);
		sigmoid_outputs_.push_back(thres_);
		sigmoid_.reset(new sigmoid_layer(context_, layer_param_));
		sigmoid_->bind(sigmoid_inputs_, sigmoid_outputs_);
		sigmoid_->init();
		sigmoid_->resize();

		rand_vec_.reset(new node_blob());
	}

	void learnable_dropout_layer::resize() {
		caffepro_layer::resize();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				rand_vec_->set_4d(nd, inputs_[0]->get(nd)->num(), 1, 1, 1, inputs_[0]->get(nd)->device_id(), context_);
			}
		}
	}

	void learnable_dropout_layer::on_before_forward() {
		sigmoid_->forward();
	}

	__global__ static void learnable_dropout_forward(const int n, const data_type* in, 
		const unsigned int* mask, const data_type *reserved_prob, const int inner_dim,
		data_type* out) {
		CUDA_KERNEL_LOOP(index, n) {
			int ins = index / inner_dim;
			out[index] = in[index] * (mask[ins] < reserved_prob[0] * UINT_MAX);
		}
	}

	void learnable_dropout_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();

		if (context_->get_phase() == caffepro_context::TRAIN) {
			CURAND_CHECK(curandGenerate(
				context_->get_current_device()->curand_handle(),
				reinterpret_cast<unsigned int *>(rand_vec_->get(device_index)->mutable_gpu_data()),
				inputs_[0]->get(device_index)->num()
				));

			KERNEL_CALL(learnable_dropout_forward, count)(
				count,
				bottom_data,
				reinterpret_cast<const unsigned int *>(rand_vec_->get(device_index)->gpu_data()),
				thres_->get(device_index)->gpu_data(),
				inputs_[0]->get(device_index)->inner_count(),
				top_data
				);

			CUDA_POST_KERNEL_CHECK;
		}
		else {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.scale_dev(count, thres_->get(device_index)->gpu_data(), bottom_data, top_data);
		}
	}

	__global__ static void learnable_dropout_backward(const int n, const data_type* in_diff,
		const unsigned int* mask, const data_type *reserved_prob, const int inner_dim,
		data_type* out_diff, const data_type scale_target) {
		CUDA_KERNEL_LOOP(index, n) {
			int ins = index / inner_dim;
			data_type v = in_diff[index] * (mask[ins] < reserved_prob[0] * UINT_MAX);
			if (scale_target == 0) {
				out_diff[index] = v;
			}
			else {
				out_diff[index] = out_diff[index] * scale_target + v;
			}
		}
	}

	void learnable_dropout_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
		auto &input = *inputs_[0]->get(device_index);
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		if (should_bp(bp_acts, 0)) {
			const data_type beta_acts = get_beta(clear_acts_diff, 0);
			data_type* bottom_diff = input.mutable_gpu_diff();
			const unsigned int* mask = reinterpret_cast<const unsigned int *>(rand_vec_->get(device_index)->gpu_data());
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(learnable_dropout_backward, count)(
				count, top_diff, mask, thres_->get(device_index)->gpu_data(),
				input.inner_count(), bottom_diff, beta_acts);
			CUDA_POST_KERNEL_CHECK;
		}

		if (should_bp(bp_weights, 0)) {
			data_type *multiplier = (data_type *)context_->get_current_device()->memory()->allocate(input.count() * sizeof(data_type));
			data_type *buffer = (data_type *)context_->get_current_device()->memory()->allocate(input.count() * sizeof(data_type));
			
			cublas.fill_constant(input.count(), 1.f, multiplier);
			cublas.mul(input.count(), top_diff, input.gpu_data(), buffer);
			cublasSetPointerMode(context_->get_current_device()->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
			cublas.dot(input.count(), buffer, multiplier, thres_->get(device_index)->mutable_gpu_diff());
			cublasSetPointerMode(context_->get_current_device()->cublas_handle(), CUBLAS_POINTER_MODE_HOST);
			//cublas.gemv(CblasNoTrans, 1, input.count(), 1.f, buffer, multiplier, 0.f, thres_->get(device_index)->mutable_gpu_diff());
			context_->get_current_device()->memory()->free(buffer);
			context_->get_current_device()->memory()->free(multiplier);
		}
	}

	void learnable_dropout_layer::backward(act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		caffepro_layer::backward(bp_acts, bp_weights, clear_acts_diff, clear_weights_diff);

		if (should_bp(bp_weights, 0)) {
			act_selector sigmoid_bp_acts = 1;
			act_selector sigmoid_clear_acts_diff = clear_weights_diff; // here, weight of this layer is the input of sigmoid layer
			sigmoid_->backward(sigmoid_bp_acts, 0, sigmoid_clear_acts_diff, 0);
		}
	}
}