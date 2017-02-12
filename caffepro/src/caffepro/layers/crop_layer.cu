
#include <caffepro/layers/crop_layer.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	crop_layer::crop_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	crop_layer::~crop_layer() {
		release_all();
	}

	void crop_layer::init() {
		check_input();

		crop_width_ = layer_param_.crop_param().crop_width();
		crop_height_ = layer_param_.crop_param().crop_height();
		stride_ = layer_param_.crop_param().stride();
	}

	void crop_layer::resize() {
		check_input();

		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				CHECK_GE(inputs_[0]->get(nd)->width(), crop_width_);
				CHECK_GE(inputs_[0]->get(nd)->height(), crop_height_);

				int output_num_mul = 0;
				if (layer_param_.crop_param().crop_type() == layer_param_.crop_param().ViewDense) {
					int width_segs = (inputs_[0]->get(nd)->width() - crop_width_) / stride_ + 1;
					int height_segs = (inputs_[0]->get(nd)->height() - crop_height_) / stride_ + 1;
					output_num_mul = width_segs * height_segs;
				}
				else if (layer_param_.crop_param().crop_type() == layer_param_.crop_param().View10) {
					output_num_mul = 10;
				}
				else {
					LOG(FATAL) << "Unknown crop type: " << layer_param_.crop_param().crop_type();
				}

				outputs_[0]->set_4d(
					nd,
					inputs_[0]->get(nd)->num() * output_num_mul,
					inputs_[0]->get(nd)->channels(),
					crop_height_,
					crop_width_,
					inputs_[0]->get(nd)->device_id(),
					context_
					);
			}
		}
	}

	__global__ static void densecrop_fwd_kernel(const int count, const int num, const int channels, const int height, const int width,
		const int crop_height, const int crop_width, const int seg_width, const int stride, const data_type *in, data_type *out) {
		CUDA_KERNEL_LOOP(index, count) {
			out += index;

			int cw = index % crop_width;
			index /= crop_width;
			int ch = index % crop_height;
			index /= crop_height;
			int c = index % channels;
			index /= channels;
			int n = index % num;
			index /= num;
			int seg_w = index % seg_width;
			int seg_h = index / seg_width;

			int w = seg_w * stride + cw;
			int h = seg_h * stride + ch;

			*out = in[((n * channels + c) * height + h) * width + w];
		}
	}

	__global__ static void crop10_fwd_kernel(const int count, const int num, const int channels, const int height, const int width,
		const int crop_height, const int crop_width, const data_type *in, data_type *out) {
		CUDA_KERNEL_LOOP(index, count) {
			out += index;

			int cw = index % crop_width;
			index /= crop_width;
			int ch = index % crop_height;
			index /= crop_height;
			int c = index % channels;
			index /= channels;
			int n = index % num;
			int vi = index / num;

			int hbase = 0, wbase = 0;
			bool flip = (vi >= 5);

			switch (vi) {
			case 0:
			case 5:
			case 2:
				wbase = 0;
				break;
			case 6:
			case 4:
			case 9:
			case 8:
				wbase = (width - crop_width) / 2;
				break;

			default:
				wbase = width - crop_width;
				break;
			}

			switch (vi) {
			case 0:
			case 6:
			case 1:
				hbase = 0;
				break;
			case 5:
			case 4:
			case 9:
			case 7:
				hbase = (height - crop_height) / 2;
				break;

			default:
				hbase = height - crop_height;
				break;
			}

			if (!flip) {
				*out = in[((n * channels + c) * height + hbase + ch) * width + wbase + cw];
			}
			else {
				*out = in[((n * channels + c) * height + hbase + ch) * width + wbase + (crop_width - cw - 1)];
			}
		}
	}

	void crop_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);

		if (layer_param_.crop_param().crop_type() == layer_param_.crop_param().ViewDense) {
			int width_segs = (input.width() - crop_width_) / stride_ + 1;
			int height_segs = (input.height() - crop_height_) / stride_ + 1;
			
			int count = width_segs * height_segs * input.num() * input.channels() * crop_height_ * crop_width_;
			KERNEL_CALL(densecrop_fwd_kernel, count)(count, input.num(), input.channels(), input.height(), input.width(), 
				crop_height_, crop_width_,
				width_segs, stride_, input.gpu_data(), output.mutable_gpu_data());

			CUDA_POST_KERNEL_CHECK;
		}
		else if (layer_param_.crop_param().crop_type() == layer_param_.crop_param().View10) {
			int count = output.count();
			KERNEL_CALL(crop10_fwd_kernel, count)(count, input.num(), input.channels(), input.height(), input.width(), crop_height_, crop_width_,
				input.gpu_data(), output.mutable_gpu_data());

			CUDA_POST_KERNEL_CHECK;
		}
		else {
			NOT_IMPLEMENTED;
		}
	}

	void crop_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		NOT_IMPLEMENTED;
	}
}