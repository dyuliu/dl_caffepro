
#include <caffepro/layers/l1norm_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	l1norm_layer::l1norm_layer(caffepro_context *context, const LayerParameter &param)
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

	l1norm_layer::~l1norm_layer() {
		release_all();
	}

	void l1norm_layer::init() {
		check_input();

		eps_ = 1e-9f;
		norm1_.reset(new node_blob());
		sum_multiplier_.reset(new node_blob());
	}

	void l1norm_layer::resize() {
		caffepro_layer::resize();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);

			if (input.reshaped()) {
				norm1_->set_4d(nd, input.num(), 1, 1, 1, input.device_id(), context_);
				sum_multiplier_->set_4d(nd, 1, input.channels(), input.height(), input.width(), input.device_id(), context_);
				sum_multiplier_->get(nd)->fill_data(1.f);
			}
		}
	}

	__global__ static void l1norm_forward(const int n, const int inner_count,
		const data_type *bottom_data, const data_type *norm1_data, data_type *top_data) {

		CUDA_KERNEL_LOOP(index, n) {
			const int num = index / inner_count;
			top_data[index] = bottom_data[index] / norm1_data[num];
		}
	}

	__global__ static void eltabs(const int n,
		const data_type *input_data, data_type *output_data) {

		CUDA_KERNEL_LOOP(index, n) {
			output_data[index] = fabsf(input_data[index]);
		}
	}

	// w = v / |v|
	void l1norm_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		KERNEL_CALL(eltabs, input.count())(input.count(), input.gpu_data(), output.mutable_gpu_data()); // compute abs(x)
		
		CUDA_POST_KERNEL_CHECK;
		
		cublas.gemv(CblasNoTrans, input.num(), input.inner_count(), 1.f, output.gpu_data(), sum_multiplier_->get(device_index)->gpu_data(),
			0.f, norm1_->get(device_index)->mutable_gpu_data()); // compute sum(abs(x)) for each filter
		cublas.add_scalar(input.num(), eps_, norm1_->get(device_index)->mutable_gpu_data());
		
		KERNEL_CALL(l1norm_forward, input.count())(input.count(), input.inner_count(), input.gpu_data(), norm1_->get(device_index)->gpu_data(), 
			output.mutable_gpu_data());

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void l1norm_backward(const int n, const int inner_count,
		const data_type *top_diff, const data_type *top_data, const data_type *norm1_data,
		const data_type *sum_top_diff_top_data,
		data_type *bottom_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, n) {
			const int num = index / inner_count;

			data_type v = 0;
			if (scale_targets != 0) {
				v = bottom_diff[index] * scale_targets;
			}

			/*bottom_diff[index] = top_diff[index] / norm1_data[num]
				- top_data[index] * sum_top_diff_top_data[num] / norm1_data[num]
				+ v;*/
			bottom_diff[index] = top_diff[index] / norm1_data[num]
				- (top_data[index] > 0 ? 1 : (top_data[index] < 0 ? -1 : 0)) * sum_top_diff_top_data[num] / norm1_data[num]
				+ v; // not differentiable at 0
		}
	}

	// d/dv = d/dw (I / |v| - 1/|v|^2 v . sign(v)^T)
	void l1norm_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			auto &input = *inputs_[0]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);

			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			data_type *buffer = (data_type *)context_->get_current_device()->memory()->allocate(input.count() * sizeof(data_type));
			
			cublas.mul(input.count(), output.gpu_diff(), output.gpu_data(), buffer); // d/dw . w
			
			cublas.gemv(CblasNoTrans, input.num(), input.inner_count(), 1.f, buffer, sum_multiplier_->get(device_index)->gpu_data(),
				0.f, norm1_->get(device_index)->mutable_gpu_diff()); // sum(d/dw . w)
			context_->get_current_device()->memory()->free(buffer);

			const data_type beta = get_beta(clear_acts_diff, 0);
			KERNEL_CALL(l1norm_backward, input.count())(input.count(), input.inner_count(), output.gpu_diff(), output.gpu_data(), norm1_->get(device_index)->gpu_data(),
				norm1_->get(device_index)->gpu_diff(), input.mutable_gpu_diff(), beta);

			CUDA_POST_KERNEL_CHECK;
		}
	}
}