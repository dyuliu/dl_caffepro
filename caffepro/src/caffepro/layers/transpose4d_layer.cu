
#include <caffepro/layers/transpose4d_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	transpose4d_layer::transpose4d_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_4D
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	transpose4d_layer::~transpose4d_layer() {
		release_all();
	}

	void transpose4d_layer::init() {
		check_input();

	}

	void transpose4d_layer::resize() {
		check_input();

		bool init = (outputs_[0]->size() == 0);
		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);
			
			if (input.reshaped()) {
				if (init) {
					width_ = inputs_[0]->get(nd)->dim_at(0);
					height_ = inputs_[0]->get(nd)->dim_at(1);
					channels_in_ = inputs_[0]->get(nd)->dim_at(2);
					channels_out_ = inputs_[0]->get(nd)->dim_at(3);

					tr_width_ = width_;
					tr_height_ = height_;
					tr_channels_in_ = channels_out_;
					tr_channels_out_ = channels_in_;

					outputs_[0]->set_4d(nd,
						tr_channels_out_, tr_channels_in_, tr_height_, tr_width_,
						inputs_[0]->get(nd)->device_id(),
						context_);
				}
				else {
					NOT_IMPLEMENTED;
				}
			}
		}
	}

	__global__ static void transpose4d_fw(const int n, const int channels_out, const int channels_in, const int height, const int width,
		const data_type *bottom_data,
		data_type *top_data) {

		CUDA_KERNEL_LOOP(index, n) {
			const int x = index % width;
			const int y = (index / width) % height;
			const int c_in = (index / width / height) % channels_in;
			const int c_out = index / width / height / channels_in;

			const int tr_x = width - x - 1;  // spatial: flip
			const int tr_y = height - y - 1; // spatial: flip
			const int tr_c_in = c_out; // channel: swap
			const int tr_c_out = c_in; // channel: swap

			top_data[((tr_c_out * channels_out + tr_c_in) * height + tr_y) * width + tr_x] = bottom_data[index];
		}
	}

	__global__ static void transpose4d_bw(const int n, const int channels_out, const int channels_in, const int height, const int width,
		const data_type *top_diff,
		data_type *bottom_diff) {

		CUDA_KERNEL_LOOP(index, n) {
			const int x = index % width;
			const int y = (index / width) % height;
			const int c_in = (index / width / height) % channels_in;
			const int c_out = index / width / height / channels_in;

			const int tr_x = width - x - 1;  // spatial: flip
			const int tr_y = height - y - 1; // spatial: flip
			const int tr_c_in = c_out; // channel: swap
			const int tr_c_out = c_in; // channel: swap

			bottom_diff[((tr_c_out * channels_out + tr_c_in) * height + tr_y) * width + tr_x] += top_diff[index];
		}
	}

	void transpose4d_layer::on_forward(int device_index) {
		
		auto &input = *inputs_[0]->get(device_index);
		
		int count = inputs_[0]->get(device_index)->count();
		
		CHECK_EQ(inputs_[0]->get(device_index)->dim_at(0), width_);
		CHECK_EQ(inputs_[0]->get(device_index)->dim_at(1), height_);
		CHECK_EQ(inputs_[0]->get(device_index)->dim_at(2), channels_in_);
		CHECK_EQ(inputs_[0]->get(device_index)->dim_at(3), channels_out_);

		KERNEL_CALL(transpose4d_fw, count)(
			count,
			channels_out_, channels_in_, height_, width_,
			inputs_[0]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data()
			);

		CUDA_POST_KERNEL_CHECK;
	}

	void transpose4d_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			auto &input = *inputs_[0]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);
			data_type beta = get_beta(clear_acts_diff, 0);
			if (beta == 0) {
				input.fill_diff(0.f); // clear diff
			}
			
			int count = inputs_[0]->get(device_index)->count();
			
			CHECK_EQ(outputs_[0]->get(device_index)->dim_at(0), tr_width_);
			CHECK_EQ(outputs_[0]->get(device_index)->dim_at(1), tr_height_);
			CHECK_EQ(outputs_[0]->get(device_index)->dim_at(2), tr_channels_in_);
			CHECK_EQ(outputs_[0]->get(device_index)->dim_at(3), tr_channels_out_);

			KERNEL_CALL(transpose4d_bw, count)(
				count,
				tr_channels_out_, tr_channels_in_, tr_height_, tr_width_,
				outputs_[0]->get(device_index)->gpu_diff(),
				inputs_[0]->get(device_index)->mutable_gpu_diff()
				);

			CUDA_POST_KERNEL_CHECK;
		}
	}
}