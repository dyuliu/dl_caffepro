
#include <caffepro/layers/dimshuffle_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	dimshuffle_layer::dimshuffle_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_NDIM
			);
	}

	dimshuffle_layer::~dimshuffle_layer() {
		release_all();
	}

	void dimshuffle_layer::init() {
		check_input();

		const int channels = inputs_[0]->dim_at(2);
		boost::shared_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, channels, 1, 1));

		weights_.resize(1);
		
		weights_[0].reset(new node_blob());
		weights_[0]->add_like(context_, *weight_template, *inputs_[0]);

		std::vector<int> dim_indexes;
		dim_indexes.resize(inputs_[0]->get(0)->channels());

		for (int i = 0; i < inputs_[0]->get(0)->channels(); i++) dim_indexes[i] = i;
		std::random_shuffle(dim_indexes.begin(), dim_indexes.end());

		for (int i = 0; i < inputs_[0]->get(0)->channels(); i++)
			weights_[0]->get(0)->mutable_cpu_data()[i] = (data_type)dim_indexes[i];
		weights_[0]->broadcast_data_via_gpu(0);
	}

	void dimshuffle_layer::resize() {
		check_input();

		bool init = (outputs_[0]->size() == 0);
		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);

			if (input.reshaped()) {
				if (init) {
					outputs_[0]->set_4d(nd,
						inputs_[0]->get(nd)->num(), inputs_[0]->get(nd)->channels(), inputs_[0]->get(nd)->height(), inputs_[0]->get(nd)->width(),
						inputs_[0]->get(nd)->device_id(),
						context_);
				}
				else {
					NOT_IMPLEMENTED;
				}
			}
		}
	}

	__global__ static void dimshuffle_fw(const int n,
		const int num, const int channels, const int height, const int width,
		const data_type *dim_indexes,
		const data_type *bottom_data,
		data_type *top_data) {

		CUDA_KERNEL_LOOP(index_top, n) {
			const int x = index_top % width;
			const int y = (index_top / width) % height;
			const int c_out = (index_top / width / height) % channels;
			const int k = index_top / width / height / channels;

			top_data[index_top] = bottom_data[((k * channels + (int)dim_indexes[c_out]) * height + y) * width + x];
		}
	}

	void dimshuffle_layer::on_forward(int device_index) {
				
			auto &output = *outputs_[0]->get(device_index);

			KERNEL_CALL(dimshuffle_fw, output.count())(
				output.count(),
				output.num(), output.channels(), output.height(), output.width(),
				weights_[0]->get(device_index)->gpu_data(),
				inputs_[0]->get(device_index)->gpu_data(),
				output.mutable_gpu_data()
				);
			CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void dimshuffle_bw(const int n,
		const int num, const int channels, const int height, const int width,
		const data_type* dim_indexes,
		const data_type *top_diff,
		data_type *bottom_diff) {

		CUDA_KERNEL_LOOP(index_top, n) {
			const int x = index_top % width;
			const int y = (index_top / width) % height;
			const int c_out = (index_top / width / height) % channels;
			const int k = index_top / width / height / channels;

			bottom_diff[((k * channels + (int)dim_indexes[c_out]) * height + y) * width + x] += top_diff[index_top];
		}
	}

	void dimshuffle_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		
		if (should_bp(bp_acts, 0)) {
			auto &input = *inputs_[0]->get(device_index);
						
			data_type beta = get_beta(clear_acts_diff, 0);
			if (beta == 0) {
				input.fill_diff(0.f); // clear diff
			}

			auto &output = *outputs_[0]->get(device_index);

			KERNEL_CALL(dimshuffle_bw, output.count())(
				output.count(),
				output.num(), output.channels(), output.height(), output.width(),
				weights_[0]->get(device_index)->gpu_data(),
				output.gpu_diff(),
				input.mutable_gpu_diff()
				);
			CUDA_POST_KERNEL_CHECK;

			weights_[0]->get(device_index)->fill_diff(0.f); // for safety
		}
	}
}