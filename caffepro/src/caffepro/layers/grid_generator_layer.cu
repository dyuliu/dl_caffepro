
#include <caffepro/layers/grid_generator_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	grid_generator_layer::grid_generator_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = 1;
		attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 2;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	grid_generator_layer::~grid_generator_layer() {
		release_all();
	}

	void grid_generator_layer::init() {
		check_input();

		method_ = layer_param_.grid_generator_param().method();
		scale_w_ = layer_param_.grid_generator_param().scale_width();
		scale_h_ = layer_param_.grid_generator_param().scale_height();

		buffer_.reset(new node_blob());
		sum_multiplier_.reset(new node_blob());
		buffer_->set_attr(node_blob::NF_TEMP);

		internal_weights_.push_back(buffer_);
	}

	void grid_generator_layer::resize() {
		check_input();

		if (inputs_[0]->reshaped() || inputs_.size() > 1 && inputs_[1]->reshaped()) {

			bool init_output = outputs_[0]->size() == 0;
			int n_device = (int)inputs_[0]->size();

			for (int nd = 0; nd < n_device; nd++) {
				auto grid_size_type = layer_param_.grid_generator_param().grid_size();
				int output_width = 0, output_height = 0;
				if (grid_size_type == GridGeneratorParameter_GridSizeOption_ABSOLUTE) {
					output_width = (int)(scale_w_ + 0.5);
					output_height = (int)(scale_h_ + 0.5);
				}
				else if (grid_size_type == GridGeneratorParameter_GridSizeOption_RELATIVE) {
					CHECK_EQ(inputs_.size(), 2);
					CHECK_EQ(inputs_[1]->get(nd)->ndim(), 4);

					output_width = (int)(scale_w_ * inputs_[1]->get(nd)->width() + 0.5);
					output_height = (int)(scale_h_ * inputs_[1]->get(nd)->height() + 0.5);
				}

				CHECK_GT(output_width, 0);
				CHECK_GT(output_height, 0);

				int param_ndim = 0;
				if (method_ == "translation-similar") {
					param_ndim = 3;
				}
				else {
					NOT_IMPLEMENTED;
				}

				CHECK_GT(param_ndim, 0);
				int inner_count = inputs_[0]->get(nd)->inner_count();
				CHECK_EQ(inner_count % param_ndim, 0);
				int output_channels = inner_count / param_ndim;

				if (init_output) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, inputs_[0]->get(nd)->num(), output_channels, output_height, output_width, inputs_[0]->get(nd)->device_id()
						)));
					outputs_[1]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, inputs_[0]->get(nd)->num(), output_channels, output_height, output_width, inputs_[0]->get(nd)->device_id()
						)));
					buffer_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, inputs_[0]->get(nd)->num(), inner_count, output_height, output_width, inputs_[0]->get(nd)->device_id()
						)));
					sum_multiplier_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, 1, 1, output_height, output_width, inputs_[0]->get(nd)->device_id()
						)));
				}
				else {
					outputs_[0]->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), output_channels, output_height, output_width);
					outputs_[1]->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), output_channels, output_height, output_width);
					buffer_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), inner_count, output_height, output_width);
					sum_multiplier_->get(nd)->reshape_4d(1, 1, output_height, output_width);
				}
				sum_multiplier_->get(nd)->fill_data((data_type)1.f);
			}
		}
	}

	// params: (scale, shift_x, shift_y)
	__global__ void translation_similar_fwd_kernel(const int count, const int height, const int width,
		const data_type *params, data_type *output_x, data_type *output_y) {
		CUDA_KERNEL_LOOP(index, count) {
			int w = index % width;
			int h = index / width % height;
			int nc = index / width / height;

			params += nc * 3;
			output_x[index] = w * (1 + params[0]) + params[1];
			output_y[index] = h * (1 + params[0]) + params[2];
		}
	}

	void grid_generator_layer::on_forward(int device_index) {
		if (method_ == "translation-similar") {
			int count = outputs_[0]->get(device_index)->count();
			KERNEL_CALL(translation_similar_fwd_kernel, count)(
				count,
				outputs_[0]->get(device_index)->height(),
				outputs_[0]->get(device_index)->width(),
				inputs_[0]->get(device_index)->gpu_data(),
				outputs_[0]->get(device_index)->mutable_gpu_data(),		// grid x
				outputs_[1]->get(device_index)->mutable_gpu_data()		// grid y
				);
		}
		else {
			NOT_IMPLEMENTED;
		}
	}

	__global__ void translation_similar_bwd_kernel(const int count, const int height, const int width,
		const data_type *output_x_diff, const data_type *output_y_diff,
		data_type *params_diff) {
		CUDA_KERNEL_LOOP(index, count) {
			int w = index % width;
			int h = index / width % height;
			int nc = index / width / height;
			int spatial_size = width * height;
			
			params_diff += nc * 3 * spatial_size + h * width + w;
			params_diff[0] = w * output_x_diff[index] + h * output_y_diff[index];
			params_diff[spatial_size] = output_x_diff[index];
			params_diff[spatial_size * 2] = output_y_diff[index];
		}
	}

	void grid_generator_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (inputs_.size() > 1 && should_bp(bp_acts, 1) && get_beta(clear_acts_diff, 1) == 0) {
			inputs_[1]->get(device_index)->fill_diff(0.f);
		}
		
		if (should_bp(bp_acts, 0)) {
			data_type beta = get_beta(clear_acts_diff, 0);
			int count = outputs_[0]->get(device_index)->count();

			if (method_ == "translation-similar") {
				KERNEL_CALL(translation_similar_bwd_kernel, count)(
					count,
					outputs_[0]->get(device_index)->height(),
					outputs_[0]->get(device_index)->width(),
					outputs_[0]->get(device_index)->gpu_diff(),
					outputs_[1]->get(device_index)->gpu_diff(),
					buffer_->get(device_index)->mutable_gpu_data()
					);
			}

			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.gemv(
				CblasNoTrans,
				buffer_->get(device_index)->num() * buffer_->get(device_index)->channels(),
				buffer_->get(device_index)->width() * buffer_->get(device_index)->height(),
				(data_type)1.f,
				buffer_->get(device_index)->gpu_data(),
				sum_multiplier_->get(device_index)->gpu_data(),
				beta,
				inputs_[0]->get(device_index)->mutable_gpu_diff()
				);
		}
	}
}