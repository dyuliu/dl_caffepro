
#include <caffepro/layers/dropout_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/math/cublas_debug.h>

namespace caffepro {
	dropout_layer::dropout_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			);
	}

	dropout_layer::~dropout_layer() {
		release_all();
	}

	void dropout_layer::init() {
		check_input();

		rand_vec_.reset(new node_blob());

		threshold_ = layer_param_.dropout_param().dropout_ratio();
		force_random_ = layer_param_.dropout_param().force_random();
		force_same_ = layer_param_.dropout_param().force_same();

		DCHECK(threshold_ >= 0.);
		DCHECK(threshold_ <= 1.);
		scale_ = 1.f / (1.f - threshold_);
		uint_thres_ = (unsigned int)(UINT_MAX * threshold_);
	}

	void dropout_layer::resize() {
		caffepro_layer::resize();

		if (rand_vec_->size() == 0) {
			rand_vec_->add_like(*inputs_[0]);
		}
		else {
			CHECK_EQ(inputs_[0]->size(), rand_vec_->size());
			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					rand_vec_->get(nd)->reshape_like(*inputs_[0]->get(nd));
				}
			}
		}
	}

	__global__ static void dropout_forward(const int n, const data_type* in,
		const unsigned int* mask, const unsigned int threshold, const float scale,
		data_type* out) {
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = in[index] * (mask[index] > threshold);
		}
	}

	__global__ static void set_int(int n, unsigned int v, unsigned int *output) {
		CUDA_KERNEL_LOOP(index, n) {
			output[index] = v;
		}
	}

	void dropout_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();

		if (context_->get_phase() == caffepro_context::TRAIN || force_random_) {
			if (force_same_) {
			
#if _DEBUG
				const int n = 100000;
				int n_open = 0;
				for (int i = 0; i < n; i++)
				{
					unsigned int r = (unsigned int)((double)rand() / RAND_MAX * UINT_MAX);
					if (r > uint_thres_)
						n_open++;
				}
				//printf("\n%s: ratio: %.5f\n", layer_param_.name().c_str(), n_open / (float)n);
#endif

				unsigned int r = (unsigned int)((double)rand() / RAND_MAX * UINT_MAX);
				KERNEL_CALL(set_int, count)(
					count,
					r,
					reinterpret_cast<unsigned int *>(rand_vec_->get(device_index)->mutable_gpu_data())
					);
				CUDA_POST_KERNEL_CHECK;
			}
			else {
				CURAND_CHECK(curandGenerate(
					context_->get_current_device()->curand_handle(),
					reinterpret_cast<unsigned int *>(rand_vec_->get(device_index)->mutable_gpu_data()),
					count
					));
			}

			KERNEL_CALL(dropout_forward, count)(
				count, 
				bottom_data, 
				reinterpret_cast<const unsigned int *>(rand_vec_->get(device_index)->gpu_data()), 
				uint_thres_,
				scale_, 
				top_data
				);

			CUDA_POST_KERNEL_CHECK;

		}
		else {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.scale(count, (data_type)1.f / scale_, bottom_data, top_data);
		}
	}

	__global__ static void dropout_backward(const int n, const data_type* in_diff,
		const unsigned int* mask, const unsigned int threshold, const float scale,
		data_type* out_diff, const data_type scale_target) {
		CUDA_KERNEL_LOOP(index, n) {
			data_type v = in_diff[index] * (mask[index] > threshold);
			if (scale_target == 0) {
				out_diff[index] = v;
			}
			else {
				out_diff[index] = out_diff[index] * scale_target + v;
			}
		}
	}

	void dropout_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			const unsigned int* mask = reinterpret_cast<const unsigned int *>(rand_vec_->get(device_index)->gpu_data());
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(dropout_backward, count)(count, top_diff, mask, uint_thres_, scale_, bottom_diff, beta_acts);
			CUDA_POST_KERNEL_CHECK;

		}
	}
}