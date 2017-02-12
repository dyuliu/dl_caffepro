
#include <caffepro/layers/flip_layer.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	flip_layer::flip_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	flip_layer::~flip_layer() {
		release_all();
	}

	void flip_layer::init() {
		check_input();
		
		keep_original_ = layer_param_.flip_param().keep_original();
	}

	void flip_layer::resize() {
		check_input();

		if (!keep_original_) {
			caffepro_layer::resize();
		}
		else {
			int n_devices = (int)inputs_[0]->size();

			for (int nd = 0; nd < n_devices; nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					auto &input = *inputs_[0]->get(nd);
					outputs_[0]->set_4d(
						nd, input.num() * 2, input.channels(), input.height(), input.width(), input.device_id(), context_
						);
				}
			}
		}
	}

	__global__ void flip_fwd_kernel(const int count, const int width, const data_type *in, data_type *out) {
		CUDA_KERNEL_LOOP(index, count) {
			int w = index % width;
			int nch = index / width;

			out[nch * width + (width - w - 1)] = in[index];
		}
	}

	void flip_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);

		data_type *output_data = output.mutable_gpu_data();
		if (keep_original_) {
			CUDA_CHECK(cudaMemcpyAsync(output_data, input.gpu_data(), input.count() * sizeof(data_type), cudaMemcpyDeviceToDevice));
			output_data += input.count();
		}

		KERNEL_CALL(flip_fwd_kernel, input.count())(input.count(), input.width(), input.gpu_data(), output_data);
	}

	void flip_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		NOT_IMPLEMENTED;
	}
}