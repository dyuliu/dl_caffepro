
#include <caffepro/layers/instance_sample_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <boost/scoped_ptr.hpp>

namespace caffepro {
	instance_sample_layer::instance_sample_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_COUNT_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_NUM_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	instance_sample_layer::~instance_sample_layer() {
		release_all();
	}

	void instance_sample_layer::init() {
		check_input();

		num_ = layer_param_.instance_sample_param().num();
		weights_.push_back(boost::shared_ptr<node_blob>(new node_blob()));

		boost::scoped_ptr<device_blob> weight_template(device_blob::create_4d(context_, num_, 1, 1, 1));
		weights_[0]->add_like(*weight_template, *inputs_[0]);
	
		// init sample
		CHECK_LE(num_, inputs_[0]->get(0)->num());

		data_type *permute = weights_[0]->get(0)->mutable_cpu_data();
		if (layer_param_.instance_sample_param().sample_method() == InstanceSampleParameter_SampleMethod_SEQ) {
			for (int i = 0; i < num_; i++) {
				permute[i] = (data_type)i;
			}
		}
		else if (layer_param_.instance_sample_param().sample_method() == InstanceSampleParameter_SampleMethod_RAND) {
			int n = inputs_[0]->get(0)->num();
			std::vector<int> p(n);
			for (int i = 0; i < n; i++) {
				p[i] = i;
			}
			std::random_shuffle(p.begin(), p.end());

			for (int i = 0; i < num_; i++) {
				permute[i] = (data_type)p[i];
			}
		}
		else {
			LOG(FATAL) << "Unknown instance sample method";
		}

		weights_[0]->get(0)->fill_diff(0.f);
		weights_[0]->broadcast_data_via_gpu(0);
		weights_[0]->broadcast_diff_via_gpu(0);
	}

	void instance_sample_layer::resize() {
		check_input();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				auto &input = *inputs_[0]->get(nd);
				CHECK_LE(num_, input.num());
				outputs_[0]->set_4d(nd, num_, input.channels(), input.height(), input.width(), input.device_id(), context_);
			}
		}
	}

	__global__ static void rand_sample_fwd_kernel(const int n, const int cols, const data_type *inputs,
		const data_type *row_index_map, data_type *outputs) {
		CUDA_KERNEL_LOOP(index, n) {
			int row = index / cols, col = index % cols;
			int src_row = (int)(row_index_map[row] + 0.5f);
			outputs[index] = inputs[src_row * cols + col];
		}
	}

	void instance_sample_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);

		if (layer_param_.instance_sample_param().sample_method() == InstanceSampleParameter_SampleMethod_SEQ) {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.copy(output.count(), input.gpu_data(), output.mutable_gpu_data());
		}
		else if (layer_param_.instance_sample_param().sample_method() == InstanceSampleParameter_SampleMethod_RAND) {
			KERNEL_CALL(rand_sample_fwd_kernel, output.count())(output.count(), input.inner_count(), input.gpu_data(), 
				weights_[0]->get(device_index)->gpu_data(), output.mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
		}
	}

	__global__ static void rand_sample_bwd_kernel(const int n, const int cols, const data_type *diff,
		const data_type *row_index_map, data_type *target_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, n) {
			int row = index / cols, col = index % cols;
			int src_row = (int)(row_index_map[row] + 0.5f);

			if (scale_targets == 0) {
				target_diff[src_row * cols + col] = diff[index];
			}
			else {
				target_diff += src_row * cols + col;
				target_diff[0] = target_diff[0] * scale_targets + diff[index];
			}
		}
	}

	void instance_sample_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		data_type beta = get_beta(clear_acts_diff, 0);

		if (beta == 0) {
			input.fill_diff(0.f);
		}

		if (layer_param_.instance_sample_param().sample_method() == InstanceSampleParameter_SampleMethod_SEQ) {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.axpby(output.count(), 1.f, output.gpu_diff(), beta, input.mutable_gpu_diff());
		}
		else if (layer_param_.instance_sample_param().sample_method() == InstanceSampleParameter_SampleMethod_RAND) {
			KERNEL_CALL(rand_sample_bwd_kernel, output.count())(output.count(), output.inner_count(), output.gpu_diff(),
				weights_[0]->get(device_index)->gpu_data(), input.mutable_gpu_diff(), beta);
			CUDA_POST_KERNEL_CHECK;
		}

		weights_[0]->get(device_index)->fill_diff(0.f);
	}
}