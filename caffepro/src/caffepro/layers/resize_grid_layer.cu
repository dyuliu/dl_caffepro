
#include <caffepro/layers/resize_grid_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	resize_grid_layer::resize_grid_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = 2;
		attr_.num_inputs_max = 3;
		attr_.num_outputs_min = 2;
		attr_.num_outputs_max = 3;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_NUM
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	resize_grid_layer::~resize_grid_layer() {
		release_all();
	}

	void resize_grid_layer::init() {
		check_input();

		if (inputs_.size() == 3) {
			CHECK_EQ(outputs_.size(), 3);
		}

		output_box_start_ = 0.f;
		output_box_scale_ = 1.f;
		if (inputs_.size() == 3) {
			output_box_start_ = config_.get<data_type>("output_box_start");
			output_box_scale_ = config_.get<data_type>("output_box_scale");
		}
		output_min_length_ = config_.get<int>("output_min_length");
		output_max_scale_ = config_.get<data_type>("output_max_scale");

		buffer_.reset(new node_blob());
		sum_multiplier_.reset(new node_blob());
		buffer_->set_attr(node_blob::NF_TEMP);

		internal_weights_.push_back(buffer_);
	}

	void resize_grid_layer::resize() {
		check_input();
		int n_devices = (int)inputs_[0]->size();
		bool init_output = (outputs_[0]->size() == 0);

		// calculate output spatial size and resize outputs
		for (int nd = 0; nd < n_devices; nd++) {
			CHECK_EQ(inputs_[1]->get(nd)->inner_count(), 1);

			int output_width, output_height;
			get_output_size(nd, output_width, output_height);

			if (init_output) {
				outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
					context_, inputs_[0]->get(nd)->num(), 1, output_height, output_width, inputs_[0]->get(nd)->device_id()
					)));
				outputs_[1]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
					context_, inputs_[0]->get(nd)->num(), 1, output_height, output_width, inputs_[0]->get(nd)->device_id()
					)));
				buffer_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
					context_, inputs_[0]->get(nd)->num(), 1, output_height, output_width, inputs_[0]->get(nd)->device_id()
					)));
				sum_multiplier_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
					context_, 1, 1, output_height, output_width, inputs_[0]->get(nd)->device_id()
					)));
			}
			else {
				outputs_[0]->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), 1, output_height, output_width);
				outputs_[1]->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), 1, output_height, output_width);
				buffer_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), 1, output_height, output_width);
				sum_multiplier_->get(nd)->reshape_4d(1, 1, output_height, output_width);
			}

			sum_multiplier_->get(nd)->fill_data(1.f);
		}

		if (inputs_.size() == 3) {
			for (int nd = 0; nd < n_devices; nd++) {
				CHECK_EQ(inputs_[2]->get(nd)->inner_count(), 4);
			}

			if (init_output) {
				outputs_[2]->add_like(context_, *inputs_[2]);
			}
			else {
				for (int nd = 0; nd < n_devices; nd++) {
					if (inputs_[2]->get(nd)->reshaped()) {
						outputs_[2]->get(nd)->reshape_like(*inputs_[2]->get(nd));
					}
				}
			}
		}
	}

	void resize_grid_layer::get_output_size(int device_index, int &output_width, int &output_height) {
		device_blob &input = *inputs_[0]->get(device_index);
		device_blob &input_scale = *inputs_[1]->get(device_index);

		int width = input.width(), height = input.height();
		double ratio = (double)width / height;
		int max_height = (int)ceil(sqrt(output_max_scale_ * output_max_scale_ / ratio));
		int max_width = (int)ceil(max_height * ratio);

		output_width = output_min_length_;
		output_height = output_min_length_;
		const data_type *learnt_scales = input_scale.cpu_data();
		int num = input.num();
		for (int i = 0; i < num; i++) {
			output_width = std::max(output_width, (int)ceil((double)width / learnt_scales[i]));
			output_height = std::max(output_height, (int)ceil((double)height / learnt_scales[i]));
		}

		output_width = std::min(output_width, max_width);
		output_height = std::min(output_height, max_height);
	}

	__global__ static void fwd_kernel(const int count, const int input_height, const int input_width,
		const int output_height, const int output_width,
		const data_type *params, data_type *output_x, data_type *output_y) {
		CUDA_KERNEL_LOOP(index, count) {
			int w = index % output_width;
			int h = index / output_width % output_height;
			int n = index / output_width / output_height;

			data_type scale = params[n];
			data_type mid_input_w = (data_type)(input_width - 1) / 2, mid_input_h = (data_type)(input_height - 1) / 2;
			data_type mid_output_w = (data_type)(output_width - 1) / 2, mid_output_h = (data_type)(output_height - 1) / 2;

			output_x[index] = (w - mid_output_w) * scale + mid_input_w;
			output_y[index] = (h - mid_output_h) * scale + mid_input_h;
		}
	}

	void resize_grid_layer::on_forward(int device_index) {
		int count = outputs_[0]->get(device_index)->count();
		int num = outputs_[0]->get(device_index)->num();

		int input_width = inputs_[0]->get(device_index)->width();
		int input_height = inputs_[0]->get(device_index)->height();
		int output_width = outputs_[0]->get(device_index)->width();
		int output_height = outputs_[0]->get(device_index)->height();

		KERNEL_CALL(fwd_kernel, count)(
			count,
			input_height,
			input_width,
			output_height,
			output_width,
			inputs_[1]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data(),		// grid x
			outputs_[1]->get(device_index)->mutable_gpu_data()		// grid y
			);

		if (inputs_.size() == 3) {
			data_type *trans_boxes = outputs_[2]->get(device_index)->mutable_cpu_data();
			const data_type *boxes = inputs_[2]->get(device_index)->cpu_data();
			const data_type *scales = inputs_[1]->get(device_index)->cpu_data();
		
			data_type mid_input_w = (data_type)(input_width - 1) / 2, mid_input_h = (data_type)(input_height - 1) / 2;
			data_type mid_output_w = (data_type)(output_width - 1) / 2, mid_output_h = (data_type)(output_height - 1) / 2;

			for (int i = 0; i < num; i++) {
				data_type *cur_trans_box = trans_boxes + 4 * i;
				const data_type *cur_box = boxes + 4 * i;
				data_type scale = scales[i];

				cur_trans_box[0] = cur_box[0] / scale + mid_output_w - mid_input_w / scale;		// left
				cur_trans_box[1] = cur_box[1] / scale + mid_output_h - mid_input_h / scale;		// top
				cur_trans_box[2] = cur_box[2] / scale + mid_output_w - mid_input_w / scale;		// right
				cur_trans_box[3] = cur_box[3] / scale + mid_output_h - mid_input_h / scale;		// bottom
			
				for (int j = 0; j < 4; j++) {
					cur_trans_box[j] = cur_trans_box[j] * output_box_scale_ + output_box_start_;
				}
			}
		}
	}

	__global__ static void bwd_kernel(const int count, const int input_height, const int input_width, 
		const int output_height, const int output_width,
		const data_type *output_x_diff, const data_type *output_y_diff,
		data_type *params_diff) {
		CUDA_KERNEL_LOOP(index, count) {
			int w = index % output_width;
			int h = index / output_width % output_height;

			data_type mid_output_w = (data_type)(output_width - 1) / 2, mid_output_h = (data_type)(output_height - 1) / 2;
			params_diff[index] = output_x_diff[index] * (w - mid_output_w) + output_y_diff[index] * (h - mid_output_h);
		}
	}

	void resize_grid_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		// we only need to bp inputs_[1]
		// fill 0 to inputs_[0] to meet the architecture requirement

		if (should_bp(bp_acts, 0) && get_beta(clear_acts_diff, 0) == 0) {
			inputs_[0]->get(device_index)->fill_diff(0.f);
		}

		if (should_bp(bp_acts, 1)) {
			data_type beta = get_beta(clear_acts_diff, 1);
			int count = outputs_[0]->get(device_index)->count();

			KERNEL_CALL(bwd_kernel, count)(
				count,
				inputs_[0]->get(device_index)->height(),
				inputs_[0]->get(device_index)->width(),
				outputs_[0]->get(device_index)->height(),
				outputs_[0]->get(device_index)->width(),
				outputs_[0]->get(device_index)->gpu_diff(),
				outputs_[1]->get(device_index)->gpu_diff(),
				buffer_->get(device_index)->mutable_gpu_data()
				);

			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.gemv(
				CblasNoTrans,
				buffer_->get(device_index)->num(), // channels == 1
				buffer_->get(device_index)->width() * buffer_->get(device_index)->height(),
				(data_type)1.f,
				buffer_->get(device_index)->gpu_data(),
				sum_multiplier_->get(device_index)->gpu_data(),
				beta,
				inputs_[1]->get(device_index)->mutable_gpu_diff()
				);
		}
	}
}