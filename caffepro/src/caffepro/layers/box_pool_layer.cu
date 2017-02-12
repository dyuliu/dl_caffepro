
#include <caffepro/layers/box_pool_layer.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	
	box_pool_layer::box_pool_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_NUM
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	box_pool_layer::~box_pool_layer() {
		release_all();
	}

	void box_pool_layer::resize() {
		check_input();

		int n_devices = (int)inputs_[0]->size();

		if (outputs_[0]->size() == 0) {
			for (int nd = 0; nd < n_devices; nd++) {
				CHECK_EQ(inputs_[1]->get(nd)->inner_count(), 4);

				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
						context_,
						inputs_[0]->get(nd)->num(),
						inputs_[0]->get(nd)->channels(),
						1,
						1,
						inputs_[0]->get(nd)->device_id()
						)
					));
			}
		}
		else {
			for (int nd = 0; nd < n_devices; nd++) {
				CHECK_EQ(inputs_[1]->get(nd)->inner_count(), 4);

				if (inputs_[0]->get(nd)->channels() != outputs_[0]->get(nd)->channels()
					|| inputs_[0]->get(nd)->num() != outputs_[0]->get(nd)->num()) {
					outputs_[0]->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), inputs_[0]->get(nd)->channels(), 1, 1);
				}
			}
		}
	}

	__global__ void box_pool_forward(const int count, const int channels, const int height, const int width,
		const data_type *feature, const data_type *boxes, data_type *output) {
		CUDA_KERNEL_LOOP(index, count) {
			int c = index % channels;
			int n = index / channels;

			boxes += n * 4;
			int left = fmaxf(0, floor(boxes[0]));
			int top = fmaxf(0, floor(boxes[1]));
			int right = fminf(width - 1, ceil(boxes[2]));
			int bottom = fminf(height - 1, ceil(boxes[3]));

			feature += (n * channels + c) * height * width;
			int pool_size = 0;
			data_type result = 0;
			for (int i = top; i <= bottom; i++) {
				for (int j = left; j <= right; j++) {
					pool_size++;
					result += feature[i * width + j];
				}
			}

			if (pool_size > 0) {
				output[index] = result / pool_size;
			}
			else {
				output[index] = 0;
			}
		}
	}

	void box_pool_layer::on_forward(int device_index) {
		auto &feature = *inputs_[0]->get(device_index);
		auto &box = *inputs_[1]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		
		KERNEL_CALL(box_pool_forward, output.count())(
			output.count(),
			feature.channels(),
			feature.height(),
			feature.width(),
			feature.gpu_data(),
			box.gpu_data(),
			output.mutable_gpu_data()
			);
	}

	__global__ void box_pool_backward(const int count, const int channels, const int height, const int width,
		const data_type *diff, const data_type *boxes, data_type *feature_diff, data_type scale_target) {
		CUDA_KERNEL_LOOP(index, count) {
			int c = index % channels;
			int n = index / channels;

			boxes += n * 4;
			int left = fmaxf(0, floor(boxes[0]));
			int top = fmaxf(0, floor(boxes[1]));
			int right = fminf(width - 1, ceil(boxes[2]));
			int bottom = fminf(height - 1, ceil(boxes[3]));

			data_type v = diff[index];
			int pool_size = 0;
			if (right >= left && bottom >= top) {
				pool_size = (right - left + 1) * (bottom - top + 1);
			}
			if (pool_size == 0) v = 0;
			else v /= pool_size;

			feature_diff += (n * channels + c) * height * width;
			for (int i = top; i <= bottom; i++) {
				for (int j = left; j <= right; j++) {
					if (scale_target == 0) {
						feature_diff[i * width + j] = v;
					}
					else {
						feature_diff[i * width + j] = feature_diff[i * width + j] * scale_target + v;
					}
				}
			}
		}
	}

	void box_pool_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			auto &feature = *inputs_[0]->get(device_index);
			auto &box = *inputs_[1]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);

			if (beta_acts == 0) {
				feature.fill_diff(0.f);
			}

			KERNEL_CALL(box_pool_backward, output.count())(
				output.count(),
				feature.channels(),
				feature.height(),
				feature.width(),
				output.gpu_diff(),
				box.gpu_data(),
				feature.mutable_gpu_diff(),
				beta_acts
				);
		}

		// do not bp inputs_[1]
	}
}