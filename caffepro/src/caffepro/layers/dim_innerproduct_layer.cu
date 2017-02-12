
#include <caffepro/layers/dim_innerproduct_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	dim_innerproduct_layer::dim_innerproduct_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_SAME_SHAPE
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM
			);
	}

	dim_innerproduct_layer::~dim_innerproduct_layer() {
		release_all();
	}

	void dim_innerproduct_layer::init() {
		check_input();

		dim_ = layer_param_.dim_innerproduct_param().dim();
	}

	void dim_innerproduct_layer::resize() {
		int n_devices = (int)inputs_[0]->size();
		bool init = (outputs_[0]->size() == 0);

		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				std::vector<int> dims = inputs_[0]->get(nd)->dims();

				CHECK_LT(dim_, dims.size());
				dims[dim_] = 1;
				if (init) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_nd(context_, (int)dims.size(), &dims[0],
						inputs_[0]->get(nd)->device_id())));
				}
				else {
					outputs_[0]->get(nd)->reshape_nd((int)dims.size(), &dims[0]);
				}
			}
		}
	}

	__global__ static void dim_innerproduct_forward(const int n, const int lower_dim, const int process_dim,
		const data_type *input1, const data_type *input2, data_type *output) {
		CUDA_KERNEL_LOOP(index, n) {
			int l = index % lower_dim; 
			int h = index / lower_dim;
			input1 += h * process_dim * lower_dim + l;
			input2 += h * process_dim * lower_dim + l;

			data_type v = 0.f;
			for (int i = 0; i < process_dim; i++) {
				int offset = i * lower_dim;
				v += input1[offset] * input2[offset];
			}
			output[index] = v;
		}
	}

	void dim_innerproduct_layer::on_forward(int device_index) {
		auto &input1 = *inputs_[0]->get(device_index);
		auto &input2 = *inputs_[1]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);

		std::vector<int> dims = input1.dims();
		int lower_dim = 1;
		for (int i = 0; i < dim_; i++) {
			lower_dim *= dims[i];
		}
		int process_dim = dims[dim_];

		KERNEL_CALL(dim_innerproduct_forward, output.count())(output.count(), lower_dim, process_dim,
			input1.gpu_data(), input2.gpu_data(), output.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void dim_innerproduct_backward(const int n, const int lower_dim, const int process_dim,
		const data_type *diff, const data_type *input1_data, const data_type *input2_data,
		data_type *input1_diff, data_type *input2_diff,
		const bool bp1, const bool bp2, const data_type scale_target1, const data_type scale_target2) {
		CUDA_KERNEL_LOOP(index, n) {
			int l = index % lower_dim;
			int h = index / lower_dim / process_dim;

			data_type d = diff[h * lower_dim + l];
			if (bp1) {
				if (scale_target1 == 0) {
					input1_diff[index] = d * input2_data[index]; // diff1 related to input2
				}
				else {
					input1_diff[index] = input1_diff[index] * scale_target1 + d * input2_data[index]; // diff1 related to input2
				}
			}

			if (bp2) {
				if (scale_target2 == 0) {
					input2_diff[index] = d * input1_data[index]; // diff2 related to input1
				}
				else {
					input2_diff[index] = input2_diff[index] * scale_target2 + d * input1_data[index]; // diff2 related to input1
				}
			}
		}
	}

	void dim_innerproduct_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0) || should_bp(bp_acts, 1)) {
			auto &input1 = *inputs_[0]->get(device_index);
			auto &input2 = *inputs_[1]->get(device_index);
			auto &output = *outputs_[0]->get(device_index);

			data_type beta1 = get_beta(clear_acts_diff, 0);
			data_type beta2 = get_beta(clear_acts_diff, 1);

			std::vector<int> dims = input1.dims();
			int lower_dim = 1;
			for (int i = 0; i < dim_; i++) {
				lower_dim *= dims[i];
			}
			int process_dim = dims[dim_];

			KERNEL_CALL(dim_innerproduct_backward, input1.count())(input1.count(), lower_dim, process_dim, output.gpu_diff(),
				input1.gpu_data(), input2.gpu_data(), input1.mutable_gpu_diff(), input2.mutable_gpu_diff(),
				should_bp(bp_acts, 0), should_bp(bp_acts, 1), beta1, beta2);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}