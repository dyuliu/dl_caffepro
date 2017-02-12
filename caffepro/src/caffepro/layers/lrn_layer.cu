
#include <caffepro/layers/lrn_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	lrn_layer::lrn_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	lrn_layer::~lrn_layer() {
		release_all();
	}

	void lrn_layer::init() {
		check_input();

		size_ = layer_param_.lrn_param().local_size();
		pre_pad_ = (size_ - 1) / 2;
		alpha_ = this->layer_param_.lrn_param().alpha();
		beta_ = this->layer_param_.lrn_param().beta();
		
		scale_.reset(new node_blob());
	}

	void lrn_layer::resize() {
		bool init = (outputs_[0]->size() == 0);

		caffepro_layer::resize();

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			if (init) {
				scale_->add_like(*inputs_[0]->get(nd));
			}
			else if (inputs_[0]->get(nd)->reshaped()) {
				scale_->get(nd)->reshape_like(*inputs_[0]->get(nd));
			}
		}
	}

	__global__ static void lrn_fill_scale(const int nthreads, const data_type* in, const int num, const int channels, const int height,
		const int width, const int size, const data_type alpha_over_size, data_type* scale) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local offset
			int w = index % width;
			int h = (index / width) % height;
			int n = index / width / height;
			int offset = (n * channels * height + h) * width + w;
			int step = height * width;
			in += offset;
			scale += offset;
			int head = 0;
			int pre_pad = (size - 1) / 2;
			int post_pad = size - pre_pad - 1;
			data_type accum_scale = 0;
			// fill the scale at [n, :, h, w]
			// accumulate values
			while (head < post_pad && head < channels) {
				accum_scale += in[head * step] * in[head * step];
				++head;
			}
			// until we reach size, nothing needs to be subtracted
			while (head < size && head < channels) {
				accum_scale += in[head * step] * in[head * step];
				scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
				++head;
			}
			// both add and subtract
			while (head < channels) {
				accum_scale += in[head * step] * in[head * step];
				accum_scale -= in[(head - size) * step] * in[(head - size) * step];
				scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
				++head;
			}
			// subtract only
			while (head < channels + post_pad) {
				if (head >= size) {
					accum_scale -= in[(head - size) * step] * in[(head - size) * step];
				}
				if (head >= post_pad) {
					scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
				}
				++head;
			}
		}
	}

	__global__ static void lrn_compute_output(const int nthreads, const data_type* in,
		const data_type* scale, const data_type negative_beta, data_type* out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			out[index] = in[index] * pow(scale[index], negative_beta);
		}
	}

	void lrn_layer::on_forward(int device_index) {
		const data_type* input_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* output_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		data_type* scale_data = scale_->get(device_index)->mutable_gpu_data();

		auto &inputs = *inputs_[0]->get(device_index);

		// First, compute scale
		// We will launch one kernel for each pixel location, and have the kernel
		// go through all the channels.
		int n_threads = inputs_[0]->get(device_index)->count() / inputs_[0]->get(device_index)->channels();
		KERNEL_CALL(lrn_fill_scale, n_threads)(
			n_threads, 
			input_data, 
			inputs.num(), 
			inputs.channels(), 
			inputs.height(), 
			inputs.width(), 
			size_,
			alpha_ / size_, 
			scale_data
			);

		// then forward
		n_threads = inputs_[0]->get(device_index)->count();
		KERNEL_CALL(lrn_compute_output, n_threads)(n_threads, input_data, scale_data, -beta_, output_data);

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void lrn_compute_diff(const int nthreads, const data_type* bottom_data, const data_type* top_data, const data_type* scale, const data_type* top_diff,
		const int num, const int channels, const int height, const int width, const int size, const data_type negative_beta,
		const data_type cache_ratio, data_type* bottom_diff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local offset
			int w = index % width;
			int h = (index / width) % height;
			int n = index / width / height;
			int offset = (n * channels * height + h) * width + w;
			int step = height * width;
			bottom_data += offset;
			top_data += offset;
			scale += offset;
			top_diff += offset;
			bottom_diff += offset;
			int head = 0;
			int pre_pad = size - (size + 1) / 2;
			int post_pad = size - pre_pad - 1;
			data_type accum_ratio = 0;
			// accumulate values
			while (head < post_pad && head < channels) {
				accum_ratio += top_diff[head * step] * top_data[head * step] /
					scale[head * step];
				++head;
			}
			// until we reach size, nothing needs to be subtracted
			while (head < size && head < channels) {
				accum_ratio += top_diff[head * step] * top_data[head * step] /
					scale[head * step];
				bottom_diff[(head - post_pad) * step] = bottom_diff[(head - post_pad) * step] * scale_targets
					+ top_diff[(head - post_pad) * step]
					* pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
					bottom_data[(head - post_pad) * step] * accum_ratio;
				++head;
			}
			// both add and subtract
			while (head < channels) {
				accum_ratio += top_diff[head * step] * top_data[head * step] /
					scale[head * step];
				accum_ratio -= top_diff[(head - size) * step] *
					top_data[(head - size) * step] / scale[(head - size) * step];
				bottom_diff[(head - post_pad) * step] = bottom_diff[(head - post_pad) * step] * scale_targets
					+ top_diff[(head - post_pad) * step]
					* pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
					bottom_data[(head - post_pad) * step] * accum_ratio;
				++head;
			}
			// subtract only
			while (head < channels + post_pad) {
				if (head >= size) {
					accum_ratio -= top_diff[(head - size) * step] *
						top_data[(head - size) * step] / scale[(head - size) * step];
				}
				if (head >= post_pad) {
					bottom_diff[(head - post_pad) * step] = bottom_diff[(head - post_pad) * step] * scale_targets
						+ top_diff[(head - post_pad) * step]
						* pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
						bottom_data[(head - post_pad) * step] * accum_ratio;
				}
				++head;
			}
		}
	}

	void lrn_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			auto &inputs = *inputs_[0]->get(device_index);
			int n_threads = inputs.count() / inputs.channels();

			if (beta_acts == 0) {
				inputs.fill_diff(0.f);
			}

			KERNEL_CALL(lrn_compute_diff, n_threads)(
				n_threads, 
				inputs.gpu_data(), 
				outputs_[0]->get(device_index)->gpu_data(),
				scale_->get(device_index)->gpu_data(), 
				outputs_[0]->get(device_index)->gpu_diff(), 
				inputs.num(), 
				inputs.channels(), 
				inputs.height(), 
				inputs.width(),
				size_, 
				-beta_, 
				(data_type)(2. * alpha_ * beta_ / size_),
				inputs.mutable_gpu_diff(),
				beta_acts
				);

			CUDA_POST_KERNEL_CHECK;
		}
	}
}