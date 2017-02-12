
#include <caffepro/layers/local_norm_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	local_norm_layer::local_norm_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	local_norm_layer::~local_norm_layer() {
		release_all();
	}

	void local_norm_layer::init() {
		check_input();

		eps_ = 1e-9f;
		EX_.reset(new node_blob());
		VarX_.reset(new node_blob());
	}

	void local_norm_layer::resize() {
		caffepro_layer::resize();
	
		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);
			
			EX_->set_4d(
				nd,
				input.num(),
				1,
				input.height(),
				input.width(),
				input.device_id(),
				context_
				);

			VarX_->set_4d(
				nd,
				input.num(),
				1,
				input.height(),
				input.width(),
				input.device_id(),
				context_
				);
		}
	}

	__global__ static void calc_crosschannel_mean_and_var(const int count, const int channels, const int spatial_size,
		const data_type *inputs, data_type *ex, data_type *var, const data_type eps) {
		CUDA_KERNEL_LOOP(index, count) {
			int hw = index % spatial_size;
			int n = index / spatial_size;

			inputs += n * channels * spatial_size + hw;
			data_type sum = 0.f, sum2 = 0.f;
			for (int i = 0; i < channels; i++) {
				data_type v = inputs[i * spatial_size];
				sum += v;
				sum2 += v * v;
			}

			sum /= channels;
			sum2 /= channels;
			ex[index] = sum;
			var[index] = sqrtf(fmaxf(sum2 - sum * sum, 0.f) + eps);
		}
	}

	__global__ static void get_crosschannel_norm(const int count, const int channels, const int spatial_size,
		const data_type *inputs, const data_type *ex, const data_type *var, data_type *outputs) {
		CUDA_KERNEL_LOOP(index, count) {
			int hw = index % spatial_size;
			int n = index / spatial_size / channels;

			int off = n * spatial_size + hw;
			data_type mean = ex[off], std = var[off];
			outputs[index] = (inputs[index] - mean) / std;
		}
	}

	void local_norm_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		auto &EX = *EX_->get(device_index);
		auto &VarX = *VarX_->get(device_index);

		int count = EX.count();
		KERNEL_CALL(calc_crosschannel_mean_and_var, count)(
			count,
			input.channels(),
			input.height() * input.width(),
			input.gpu_data(),
			EX.mutable_gpu_data(),
			VarX.mutable_gpu_data(),
			eps_
			);

		count = output.count();
		KERNEL_CALL(get_crosschannel_norm, count)(
			count,
			input.channels(),
			input.height() * input.width(),
			input.gpu_data(),
			EX.gpu_data(),
			VarX.gpu_data(),
			output.mutable_gpu_data()
			);

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void prepare_backward(const int count, const int channels, const int spatial_size,
		const data_type *diff, const data_type *fwd_outputs, const data_type *var,
		data_type *avg_diff, data_type *avg_prod_diff) {
		CUDA_KERNEL_LOOP(index, count) {
			int hw = index % spatial_size;
			int n = index / spatial_size;

			int off = n * channels * spatial_size + hw;
			diff += off;
			fwd_outputs += off;
			data_type sum_diff = 0.f, sum_prod_diff = 0.f;
			for (int i = 0; i < channels; i++) {
				sum_diff += diff[i * spatial_size];
				sum_prod_diff += diff[i * spatial_size] * fwd_outputs[i * spatial_size];
			}

			avg_diff[index] = sum_diff / channels;
			avg_prod_diff[index] = sum_prod_diff / channels;
		}
	}

	__global__ static void calc_bp(const int count, const int channels, const int spatial_size,
		const data_type *diff, const data_type *fwd_outputs, 
		const data_type *var, const data_type *avg_diff, const data_type *avg_prod_diff,
		data_type *target_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, count) {
			int hw = index % spatial_size;
			int n = index / spatial_size / channels;

			int off = n * spatial_size + hw;
			data_type v = diff[index] - avg_diff[off] - fwd_outputs[index] * avg_prod_diff[off];
			v /= var[off];

			if (scale_targets == 0) {
				target_diff[index] = v;
			}
			else {
				target_diff[index] = target_diff[index] * scale_targets + v;
			}
		}
	}

	void local_norm_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			const data_type beta = get_beta(clear_acts_diff, 0);
			auto &input = *inputs_[0]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);
			
			const data_type *var_data = VarX_->get(device_index)->gpu_data();
			data_type *avg_diff_buffer = EX_->get(device_index)->mutable_gpu_diff(); // hack: use EX_diff as buffer
			data_type *avg_diff_prod_buffer = VarX_->get(device_index)->mutable_gpu_diff(); // hack: use VarX_diff as buffer

			int count = VarX_->get(device_index)->count();
			KERNEL_CALL(prepare_backward, count)(
				count,
				input.channels(), 
				input.height() * input.width(),
				output.gpu_diff(),
				output.gpu_data(),
				var_data,
				avg_diff_buffer,
				avg_diff_prod_buffer
				);

			count = input.count();
			KERNEL_CALL(calc_bp, count)(
				count,
				input.channels(),
				input.height() * input.width(),
				output.gpu_diff(),
				output.gpu_data(),
				var_data,
				avg_diff_buffer,
				avg_diff_prod_buffer,
				input.mutable_gpu_diff(),
				beta
				);

			CUDA_POST_KERNEL_CHECK;
		}
	}
}