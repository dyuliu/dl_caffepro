
#include <caffepro/layers/slice_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>


namespace caffepro {
	slice_layer::slice_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = 1;
		attr_.num_outputs_max = INT_MAX;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_NDIM
			);
	}

	slice_layer::~slice_layer() {
		release_all();
	}

	void slice_layer::init() {
		check_input();

		num_slices_ = outputs_.size();

		CHECK_EQ(inputs_[0]->get(0)->channels() % num_slices_, 0);
		channels_slices_ = inputs_[0]->get(0)->channels() / num_slices_;

	}

	void slice_layer::resize() {
		check_input();

		bool init = (outputs_[0]->size() == 0);
		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);

			if (input.reshaped()) {
				if (init) {
					for (int i = 0; i < num_slices_; i++) {
						outputs_[i]->set_4d(nd,
							inputs_[0]->get(nd)->num(), channels_slices_, inputs_[0]->get(nd)->height(), inputs_[0]->get(nd)->width(),
							inputs_[0]->get(nd)->device_id(),
							context_);
					}
				}
				else {
					NOT_IMPLEMENTED;
				}
			}
		}
	}

	__global__ static void slice_fw(const int n,
		const int num, const int channels_out, const int height, const int width,
		const int channels_in,
		const int channels_in_offset,
		const data_type *bottom_data,
		data_type *top_data) {

		CUDA_KERNEL_LOOP(index_top, n) {
			const int x = index_top % width;
			const int y = (index_top / width) % height;
			const int c_out = (index_top / width / height) % channels_out;
			const int k = index_top / width / height / channels_out;

			top_data[index_top] = bottom_data[((k * channels_in + c_out + channels_in_offset) * height + y) * width + x];
		}
	}

	void slice_layer::on_forward(int device_index) {
		
		for (int i = 0; i < num_slices_; i++) {
			
			auto &output = *outputs_[i]->get(device_index);
			
			KERNEL_CALL(slice_fw, output.count())(
				output.count(),
				output.num(), channels_slices_, output.height(), output.width(),
				inputs_[0]->get(device_index)->channels(),
				i * channels_slices_,
				inputs_[0]->get(device_index)->gpu_data(),
				output.mutable_gpu_data()
				);
			CUDA_POST_KERNEL_CHECK;
		} //i

	}

	__global__ static void slice_bw(const int n,
		const int num, const int channels_out, const int height, const int width,
		const int channels_in,
		const int channels_in_offset,
		const data_type *top_diff,
		data_type *bottom_diff) {

		CUDA_KERNEL_LOOP(index_top, n) {
			const int x = index_top % width;
			const int y = (index_top / width) % height;
			const int c_out = (index_top / width / height) % channels_out;
			const int k = index_top / width / height / channels_out;

			bottom_diff[((k * channels_in + c_out + channels_in_offset) * height + y) * width + x] += top_diff[index_top];
		}
	}

	void slice_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		
		if (should_bp(bp_acts, 0)) {
			auto &input = *inputs_[0]->get(device_index);
						
			data_type beta = get_beta(clear_acts_diff, 0);
			if (beta == 0) {
				input.fill_diff(0.f); // clear diff
			}

			for (int i = 0; i < num_slices_; i++) {

				auto &output = *outputs_[i]->get(device_index);

				KERNEL_CALL(slice_bw, output.count())(
					output.count(),
					output.num(), channels_slices_, output.height(), output.width(),
					inputs_[0]->get(device_index)->channels(),
					i * channels_slices_,
					output.gpu_diff(),
					input.mutable_gpu_diff()
					);
				CUDA_POST_KERNEL_CHECK;
			} //i
		}
	}
}