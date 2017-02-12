
#include <caffepro/layers/dropoutsame_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/math/cublas_debug.h>

namespace caffepro {
	dropoutsame_layer::dropoutsame_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_ALLOW_INPLACE
			);
	}

	dropoutsame_layer::~dropoutsame_layer() {
		release_all();
	}

	void dropoutsame_layer::init() {
		check_input();

		threshold_ = layer_param_.dropoutsame_param().dropout_ratio();
		force_random_ = layer_param_.dropoutsame_param().force_random();

		DCHECK(threshold_ >= 0.);
		DCHECK(threshold_ <= 1.);
		
		scale_ = 1.f / (1.f - threshold_);
		uint_thres_ = (unsigned int)(UINT_MAX * threshold_);
	}

	void dropoutsame_layer::resize() {
		caffepro_layer::resize();
	}

	void dropoutsame_layer::on_before_forward() {
		// threshold_ = 0.2 means removing 20%
		unsigned int uint_r = (unsigned int)((double)rand() / RAND_MAX * UINT_MAX);
		open_or_not_ = (uint_r > uint_thres_) ? true : false;
	}

	void dropoutsame_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const int count = inputs_[0]->get(device_index)->count();

		if (context_->get_phase() == caffepro_context::TRAIN || force_random_) {
			
#if _DEBUG
			const int n = 100000;
			int n_open = 0;
			for (int i = 0; i < n; i++)
			{
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				if (r > threshold_)
					n_open ++;
			}
			printf("ratio: %.5f\n", n_open / (float)n);
#endif
				
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			if (open_or_not_)
				cublas.copy(count, bottom_data, top_data);
			else
				outputs_[0]->get(device_index)->fill_data(0.f);
				//cublas.scale(count, (data_type)0.0f, bottom_data, top_data);
			

		}
		else {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.scale(count, (data_type)1.f / scale_, bottom_data, top_data);
		}
	}

	__global__ void dropoutsame_backward(const int n, const data_type* in_diff,
		const bool open_or_not,
		data_type* out_diff, const data_type scale_target) {
		CUDA_KERNEL_LOOP(index, n) {
			data_type v = in_diff[index] * (open_or_not);
			if (scale_target == 0) {
				out_diff[index] = v;
			}
			else {
				out_diff[index] = out_diff[index] * scale_target + v;
			}
		}
	}

	void dropoutsame_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			
			const int count = inputs_[0]->get(device_index)->count();

			KERNEL_CALL(dropoutsame_backward, count)(count, top_diff, open_or_not_, bottom_diff, beta_acts);
			CUDA_POST_KERNEL_CHECK;

		}
	}
}