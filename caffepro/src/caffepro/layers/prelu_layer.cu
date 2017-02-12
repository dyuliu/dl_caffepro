
#include <caffepro/layers/prelu_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/utils/filler.h>
#include <caffepro/proto/caffe.pb.h>
#include <boost/scoped_ptr.hpp>

namespace caffepro {
	prelu_layer::prelu_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_4D
			);
	}

	prelu_layer::~prelu_layer() {
		release_all();
	}

	void prelu_layer::init() {
		check_input();

		channels_ = inputs_[0]->dim_at(2);

		boost::shared_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, channels_, 1, 1));
		weights_.push_back(boost::shared_ptr<node_blob>(new node_blob));
		weights_[0]->add_like(*weight_template, *inputs_[0]);
		
		// fill weights
		boost::shared_ptr<filler> leak_filler(filler::create(context_, layer_param_.learnable_leak_relu_param().relu_leak_param()));
		leak_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		sum_multiplier_.reset(new node_blob());
	}

	void prelu_layer::resize() {
		caffepro_layer::resize();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);

			if (input.reshaped()) {
				CHECK_EQ(channels_, input.channels());

				sum_multiplier_->set_4d(nd, input.num(), 1, input.height(), input.width(), input.device_id(), context_);
				sum_multiplier_->get(nd)->fill_data(1.f);
			}
		}
	}

	__global__ static void leak_channel_forward(const int nthreads, const int channels, const int spatial_size,
		const data_type *bottom_data, data_type *top_data, const data_type *weight_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			if (bottom_data[index] >= 0) {
				top_data[index] = bottom_data[index];
			}
			else {
				int c = (index / spatial_size) % channels;
				top_data[index] = bottom_data[index] * weight_data[c];
			}
		}
	}

	void prelu_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);

		KERNEL_CALL(leak_channel_forward, input.count())(input.count(), channels_, input.width() * input.height(), input.gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data(), weights_[0]->get(device_index)->gpu_data());

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void leak_channel_backward_act(const int nthreads, const int channels, const int height_width,
		const data_type *bottom_data, const data_type *top_diff, const data_type *weight_data,
		data_type *bottom_diff, data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			data_type v = 0;
			if (bottom_data[index] >= 0) {
				v = top_diff[index];
			}
			else {
				int c = (index / height_width) % channels;
				v = top_diff[index] * weight_data[c];
			}

			if (scale_targets == 0) {
				bottom_diff[index] = v;
			}
			else {
				bottom_diff[index] = bottom_diff[index] * scale_targets + v;
			}
		}
	}

	// buffer: (width, height, num, channels)
	__global__ static void leak_channel_backward_weight(const int nthreads, const int num, const int channels, const int height_width,
		const data_type *bottom_data, const data_type *top_diff, data_type *buffer) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int wh = index % height_width;
			int c = index / height_width % channels;
			int n = index / height_width / channels;

			data_type v = 0;
			if (bottom_data[index] < 0) {
				v = top_diff[index] * bottom_data[index];
			}
			buffer[(c * num + n) * height_width + wh] = v;
		}
	}

	void prelu_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		auto &weight = *weights_[0]->get(device_index);

		if (should_bp(bp_acts, 0)) {
			data_type beta = get_beta(clear_acts_diff, 0);
			KERNEL_CALL(leak_channel_backward_act, input.count())(input.count(), channels_, input.width() * input.height(), input.gpu_data(), output.gpu_diff(), weight.gpu_data(), 
				input.mutable_gpu_diff(), beta);
			CUDA_POST_KERNEL_CHECK;
		}

		if (should_bp(bp_weights, 0)) {
			data_type beta = get_beta(clear_weights_diff, 0);
			data_type *buffer = reinterpret_cast<data_type *>(context_->get_current_device()->memory()->allocate(input.count() * sizeof(data_type)));

			KERNEL_CALL(leak_channel_backward_weight, input.count())(input.count(), input.num(), channels_, input.height() * input.width(), input.gpu_data(), 
				output.gpu_diff(), buffer);
			CUDA_POST_KERNEL_CHECK;

			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.gemv(CblasNoTrans, channels_, input.count() / channels_, 1.f, buffer, sum_multiplier_->get(device_index)->gpu_data(), beta, weight.mutable_gpu_diff());
			
			context_->get_current_device()->memory()->free(buffer);
		}
	}
}