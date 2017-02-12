
#include <caffepro/layers/sample_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

#include <algorithm>
#include <cmath>

namespace caffepro {
	sample_layer::sample_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		// 3 inputs: image (feature map), x grid, y grid
		attr_.num_inputs_min = attr_.num_inputs_max = 3;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_NUM
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	sample_layer::~sample_layer() {
		release_all();
	}

	void sample_layer::init() {
		check_input();

		grid_buffer_.reset(new node_blob());
		grid_sum_multiplier_.reset(new node_blob());
		grid_buffer_->set_attr(node_blob::NF_TEMP);

		act_buffer_.reset(new node_blob());
		act_sum_multiplier_.reset(new node_blob());
		act_buffer_->set_attr(node_blob::NF_TEMP);

		internal_weights_.push_back(grid_buffer_);
		internal_weights_.push_back(act_buffer_);
	}

	void sample_layer::resize() {
		check_input();

		if (inputs_[0]->reshaped() || inputs_[1]->reshaped() || inputs_[2]->reshaped()) {
			bool init_output = outputs_[0]->size() == 0;
			int n_device = (int)inputs_[0]->size();

			for (int nd = 0; nd < n_device; nd++) {
				CHECK(inputs_[1]->get(nd)->same_shape(*inputs_[2]->get(nd)));

				int n_copies = inputs_[1]->get(nd)->channels();
				int image_channels = inputs_[0]->get(nd)->channels();
				int output_channels = (layer_param_.sample_param().concat_output() == SampleParameter_ConcatOutputOption_CHANNEL
					? image_channels * n_copies
					: image_channels
					);
				int output_num = (layer_param_.sample_param().concat_output() == SampleParameter_ConcatOutputOption_CHANNEL
					? inputs_[0]->get(nd)->num()
					: inputs_[0]->get(nd)->num() * n_copies
					);
				int output_width = inputs_[1]->get(nd)->width();
				int output_height = inputs_[1]->get(nd)->height();

				if (init_output) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, output_num, output_channels, output_height, output_width, inputs_[0]->get(nd)->device_id()
						)));
					grid_buffer_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, inputs_[0]->get(nd)->num(), n_copies * output_height * output_width * image_channels, 1, 1, inputs_[0]->get(nd)->device_id()
						)));
					grid_sum_multiplier_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, 1, image_channels, 1, 1, inputs_[0]->get(nd)->device_id()
						)));
					act_buffer_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, inputs_[0]->get(nd)->num(), inputs_[0]->get(nd)->inner_count() * n_copies, 1, 1, inputs_[0]->get(nd)->device_id()
						)));
					act_sum_multiplier_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						context_, 1, n_copies, 1, 1, inputs_[0]->get(nd)->device_id()
						)));
				}
				else {
					outputs_[0]->get(nd)->reshape_4d(output_num, output_channels, output_height, output_width);
					grid_buffer_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), n_copies * output_height * output_width * image_channels, 1, 1);
					grid_sum_multiplier_->get(nd)->reshape_4d(1, image_channels, 1, 1);
					act_buffer_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), inputs_[0]->get(nd)->inner_count() * n_copies, 1, 1);
					act_sum_multiplier_->get(nd)->reshape_4d(1, n_copies, 1, 1);
				}

				grid_sum_multiplier_->get(nd)->fill_data(1.f);
				act_sum_multiplier_->get(nd)->fill_data(1.f);
			}
		}
	}

	// output: (w, h, c, cpy, n)
	__global__ void sample_fwd_kernel(const int count, const int n_copies, const int height, const int width,
		const int input_channels, const int input_height, const int input_width,
		const data_type *input_image, const data_type *x_grid, const data_type *y_grid,
		data_type *output) {
		CUDA_KERNEL_LOOP(index, count) {
			const int output_index = index;

			int w = index % width;
			index /= width;
			int h = index % height;
			index /= height;
			int c = index % input_channels;
			index /= input_channels;
			int cpy = index % n_copies;
			int n = index / n_copies;

			int grid_index = ((n * n_copies + cpy) * height + h) * width + w;
			data_type xi = x_grid[grid_index];
			data_type yi = y_grid[grid_index];

			int x_start = fmaxf(floorf(xi), 0);
			int x_end = fminf(ceilf(xi), input_width - 1);
			int y_start = fmaxf(floorf(yi), 0);
			int y_end = fminf(ceilf(yi), input_height - 1);

			data_type v = 0;
			for (int i = y_start; i <= y_end; i++) {
				const data_type *input = input_image + ((n * input_channels + c) * input_height + i) * input_width;
				for (int j = x_start; j <= x_end; j++) {
					v += input[j] * (1 - fabsf(xi - j)) * (1 - fabsf(yi - i));
				}
			}

			output[output_index] = v;
		}
	}

	void sample_layer::on_forward(int device_index) {
		device_blob &input_image = *inputs_[0]->get(device_index);
		device_blob &grid_x = *inputs_[1]->get(device_index);
		device_blob &grid_y = *inputs_[2]->get(device_index);
		device_blob &output = *outputs_[0]->get(device_index);

		int count = output.count();

		KERNEL_CALL(sample_fwd_kernel, count)(
			count,
			grid_x.channels(),
			output.height(),
			output.width(),
			input_image.channels(),
			input_image.height(),
			input_image.width(),
			input_image.gpu_data(),
			grid_x.gpu_data(),
			grid_y.gpu_data(),
			output.mutable_gpu_data()
			);
	}

	// grid_buffer: (c, w, h, cpy, n)
	// output_diff: (w, h, c, cpy, n)
	__global__ void sample_bwd_grid_kernel(const int count, const int n_copies, const int height, const int width,
		const int input_channels, const int input_height, const int input_width,
		const data_type *input_image, const data_type *x_grid, const data_type *y_grid, const data_type *output_diff, 
		data_type *grid_buffer, const int bp_direction) {
		CUDA_KERNEL_LOOP(index, count) {
			const int output_index = index;

			int c = index % input_channels;
			index /= input_channels;
			int w = index % width;
			index /= width;
			int h = index % height;
			index /= height;
			int cpy = index % n_copies;
			int n = index / n_copies;

			int grid_index = ((n * n_copies + cpy) * height + h) * width + w;
			data_type xi = x_grid[grid_index];
			data_type yi = y_grid[grid_index];

			int x_start = fmaxf(floorf(xi), 0);
			int x_end = fminf(ceilf(xi), input_width - 1);
			int y_start = fmaxf(floorf(yi), 0);
			int y_end = fminf(ceilf(yi), input_height - 1);

			data_type v = 0;
			for (int i = y_start; i <= y_end; i++) {
				const data_type *input = input_image + ((n * input_channels + c) * input_height + i) * input_width;
				for (int j = x_start; j <= x_end; j++) {
					if (bp_direction == 0) { // bp grid_x
						v += input[j] * (1 - fabsf(yi - i)) * (j >= xi ? 1 : -1);
					}
					else if (bp_direction == 1) { // bp grid_y
						v += input[j] * (1 - fabsf(xi - j)) * (i >= yi ? 1 : -1);
					}
				}
			}

			output_diff += (((n * n_copies + cpy) * input_channels + c) * height + h) * width + w;
			grid_buffer[output_index] = v * output_diff[0];
		}
	}

	// backtrack_index_start: (iw, ih, cpy, n)
	// x_grid, y_grid: (w, h, cpy, n)
	// output_diff: (w, h, c, cpy, n)
	// act_buffer: (cpy, iw, ih, c, n)
	__global__ void sample_bwd_act_kernel(const int count, const int n_copies, const int height, const int width,
		const int input_channels, const int input_height, const int input_width,
		const data_type *x_grid, const data_type *y_grid, const data_type *output_diff,
		const int *backtrack_indexes, const int *backtrack_index_start,
		data_type *act_buffer) {
		CUDA_KERNEL_LOOP(index, count) {
			const int output_index = index;

			int cpy = index % n_copies;
			index /= n_copies;
			int iw = index % input_width;
			index /= input_width;
			int ih = index % input_height;
			index /= input_height;
			int c = index % input_channels;
			int n = index / input_channels;

			backtrack_indexes += backtrack_index_start[((n * n_copies + cpy) * input_height + ih) * input_width + iw];
			int n_related = *backtrack_indexes;
			backtrack_indexes++;

			output_diff += ((n * n_copies + cpy) * input_channels + c) * height * width;

			data_type v = 0;
			for (int k = 0; k < n_related; k++) {
				int src_grid_index = backtrack_indexes[k];

				data_type xi = x_grid[src_grid_index];
				data_type yi = y_grid[src_grid_index];

				int src_w = src_grid_index % width;
				int src_h = src_grid_index / width % height;

				data_type diff = output_diff[src_h * width + src_w];
				v += diff * (1 - fabsf(xi - iw)) * (1 - fabsf(yi - ih));
			}

			act_buffer[output_index] = v;
		}
	}

	void sample_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		device_blob &input_image = *inputs_[0]->get(device_index);
		device_blob &grid_x = *inputs_[1]->get(device_index);
		device_blob &grid_y = *inputs_[2]->get(device_index);
		device_blob &output = *outputs_[0]->get(device_index);

		if (should_bp(bp_acts, 1)) { // bp grid_x
			data_type beta = get_beta(clear_acts_diff, 1);
			int count = grid_buffer_->get(device_index)->count();

			KERNEL_CALL(sample_bwd_grid_kernel, count)(
				count,
				grid_x.channels(),
				output.height(),
				output.width(),
				input_image.channels(),
				input_image.height(),
				input_image.width(),
				input_image.gpu_data(),
				grid_x.gpu_data(),
				grid_y.gpu_data(),
				output.gpu_diff(),
				grid_buffer_->get(device_index)->mutable_gpu_data(),
				0
				);

			cublas.gemv(
				CblasNoTrans,
				grid_x.count(),
				input_image.channels(),
				(data_type)1.f,
				grid_buffer_->get(device_index)->gpu_data(),
				grid_sum_multiplier_->get(device_index)->gpu_data(),
				beta,
				grid_x.mutable_gpu_diff()
				);
		}

		if (should_bp(bp_acts, 2)) { // bp grid_y
			data_type beta = get_beta(clear_acts_diff, 2);
			int count = grid_buffer_->get(device_index)->count();

			KERNEL_CALL(sample_bwd_grid_kernel, count)(
				count,
				grid_x.channels(), // grid_y has the same shape with grid_x
				output.height(),
				output.width(),
				input_image.channels(),
				input_image.height(),
				input_image.width(),
				input_image.gpu_data(),
				grid_x.gpu_data(),
				grid_y.gpu_data(),
				output.gpu_diff(),
				grid_buffer_->get(device_index)->mutable_gpu_data(),
				1
				);

			cublas.gemv(
				CblasNoTrans,
				grid_x.count(), // grid_y has the same shape with grid_x
				input_image.channels(),
				(data_type)1.f,
				grid_buffer_->get(device_index)->gpu_data(),
				grid_sum_multiplier_->get(device_index)->gpu_data(),
				beta,
				grid_y.mutable_gpu_diff()
				);
		}

		if (should_bp(bp_acts, 0)) {
			data_type beta = get_beta(clear_acts_diff, 0);
			
			std::vector<int> indexes, index_start;
			get_grid_backtrack_index(device_index, indexes, index_start);

			int *indexes_data = reinterpret_cast<int *>(context_->get_current_device()->memory()->allocate(indexes.size() * sizeof(int)));
			int *index_start_data = reinterpret_cast<int *>(context_->get_current_device()->memory()->allocate(index_start.size() * sizeof(int)));

			CUDA_CHECK(cudaMemcpy(indexes_data, &indexes[0], indexes.size() * sizeof(int), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(index_start_data, &index_start[0], index_start.size() * sizeof(int), cudaMemcpyHostToDevice));

			int count = act_buffer_->get(device_index)->count();
			KERNEL_CALL(sample_bwd_act_kernel, count)(
				count,
				grid_x.channels(),
				output.height(),
				output.width(),
				input_image.channels(),
				input_image.height(),
				input_image.width(),
				grid_x.gpu_data(),
				grid_y.gpu_data(),
				output.gpu_diff(),
				indexes_data,
				index_start_data,
				act_buffer_->get(device_index)->mutable_gpu_data()
				);

			cublas.gemv(
				CblasNoTrans,
				input_image.count(),
				grid_x.channels(),
				(data_type)1.f,
				act_buffer_->get(device_index)->gpu_data(),
				act_sum_multiplier_->get(device_index)->gpu_data(),
				beta,
				input_image.mutable_gpu_diff()
				);

			// although most of the operations above are asynchronous, releasing memory here is still safe
			// because there's only one stream on the device
			context_->get_current_device()->memory()->free(indexes_data);
			context_->get_current_device()->memory()->free(index_start_data);
		}
	}

	void sample_layer::get_grid_backtrack_index(int device_index, std::vector<int> &indexes, std::vector<int> &index_start) {
		device_blob &input_image = *inputs_[0]->get(device_index);
		device_blob &grid_x = *inputs_[1]->get(device_index);
		device_blob &grid_y = *inputs_[2]->get(device_index);

		int input_width = input_image.width(), input_height = input_image.height();
		int width = grid_x.width(), height = grid_x.height();
		int num = input_image.num();
		int n_copies = grid_x.channels();

		index_start.resize(num * n_copies * input_height * input_width);
		std::fill(index_start.begin(), index_start.end(), 1);

		int input_spatial_size = input_width * input_height;
		int grid_spatial_size = width * height;
		int n_grids = num * n_copies;

		// get grid cpu data
		const data_type *grid_x_data = grid_x.cpu_data();
		const data_type *grid_y_data = grid_y.cpu_data();

#pragma omp parallel for
		for (int ng = 0; ng < n_grids; ng++) {
			int grid_start = ng * grid_spatial_size;
			int result_start = ng * input_spatial_size;

			for (int s = 0; s < grid_spatial_size; s++) {
				int grid_pos = grid_start + s;
				data_type xi = grid_x_data[grid_pos];
				data_type yi = grid_y_data[grid_pos];

				int x_start = std::max((int)floorf(xi), 0);
				int x_end = std::min((int)ceilf(xi), input_width - 1);
				int y_start = std::max((int)floorf(yi), 0);
				int y_end = std::min((int)ceilf(yi), input_height - 1);
			
				for (int i = y_start; i <= y_end; i++) {
					int result_offset = result_start + i * input_width;
					for (int j = x_start; j <= x_end; j++) {
						++index_start[result_offset + j];
					}
				}
			}
		}

		int sum = 0;
		for (auto iter = index_start.begin(); iter != index_start.end(); ++iter) {
			sum += *iter;
			*iter = sum - *iter;
		}

		indexes.resize(sum);
		memset(&indexes[0], 0, sum * sizeof(int));

#pragma omp parallel for
		for (int ng = 0; ng < n_grids; ng++) {
			int grid_start = ng * grid_spatial_size;
			int result_start = ng * input_spatial_size;

			for (int s = 0; s < grid_spatial_size; s++) {
				int grid_pos = grid_start + s;
				data_type xi = grid_x_data[grid_pos];
				data_type yi = grid_y_data[grid_pos];

				int x_start = std::max((int)floorf(xi), 0);
				int x_end = std::min((int)ceilf(xi), input_width - 1);
				int y_start = std::max((int)floorf(yi), 0);
				int y_end = std::min((int)ceilf(yi), input_height - 1);
			
				for (int i = y_start; i <= y_end; i++) {
					int result_offset = result_start + i * input_width;
					for (int j = x_start; j <= x_end; j++) {
						int index_pos = index_start[result_offset + j];
						indexes[index_pos + (++indexes[index_pos])] = grid_pos;
					}
				}
			}
		}
	}
}