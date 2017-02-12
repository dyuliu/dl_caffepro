
#include <caffepro/layers/padding_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	padding_layer::padding_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	padding_layer::~padding_layer() {
		release_all();
	}

	void padding_layer::init() {
		check_input();

		pad_ = (int)layer_param_.padding_param().pad();
	}

	void padding_layer::resize() {
		check_input();

		bool init = (outputs_[0]->size() == 0);
		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);

			if (init || input.reshaped()) {
				outputs_[0]->set_4d(
					nd,
					input.num(), input.channels(), input.height() + pad_ * 2, input.width() + pad_ * 2,
					input.device_id(), context_
					);
			}
		}
	}

	__global__ static void padding_forward(const int count, const data_type* in, data_type* out,
		const int num, const int channel, const int height_in, const int width_in,
		const int pad) {
		CUDA_KERNEL_LOOP(index, count) {
			int height_out = height_in + pad + pad;
			int width_out = width_in + pad + pad;
			int w = index % width_in;
			index /= width_in;
			int h = index % height_in;
			index /= height_in;
			int c = index % channel;
			index /= channel;
			out[((index * channel + c) * height_out + h + pad) * width_out + pad + w] =
				in[((index * channel + c) * height_in + h) * width_in + w];
		}
	}

	void padding_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);

		const int count = input.count();

		// First, set all data to be zero for the boundary pixels
		output.fill_data(0.f);

		KERNEL_CALL(padding_forward, count)(
			count, input.gpu_data(), output.mutable_gpu_data(), 
			input.num(), input.channels(), input.height(), input.width(),
			pad_
			);

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void padding_backward(const int count, const data_type* in, data_type* out,
		const int num, const int channel, const int height_in, const int width_in,
		const int pad, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, count) {
			int height_out = height_in + pad + pad;
			int width_out = width_in + pad + pad;
			int w = index % width_in;
			index /= width_in;
			int h = index % height_in;
			index /= height_in;
			int c = index % channel;
			index /= channel;

			if (scale_targets == 0) {
				out[((index * channel + c) * height_in + h) * width_in + w] =
					in[((index * channel + c) * height_out + h + pad) *
					width_out + pad + w];
			}
			else {
				out[((index * channel + c) * height_in + h) * width_in + w] = 
					out[((index * channel + c) * height_in + h) * width_in + w] * scale_targets
					+ in[((index * channel + c) * height_out + h + pad) *
					width_out + pad + w];
			}
		}
	}

	void padding_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			data_type beta = get_beta(clear_acts_diff, 0);

			auto &input = *inputs_[0]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);

			const int count = input.count();

			KERNEL_CALL(padding_backward, count)(
				count, output.gpu_diff(), input.mutable_gpu_diff(), 
				input.num(), input.channels(), input.height(), input.width(),
				pad_, beta);

			CUDA_POST_KERNEL_CHECK;
		}
	}
}