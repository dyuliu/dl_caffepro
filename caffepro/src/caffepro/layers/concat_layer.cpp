
#include <caffepro/layers/concat_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	using std::vector;

	concat_layer::concat_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = 1;
		attr_.num_inputs_max = INT_MAX;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_NDIM
			| layer_attribute::CF_REQUIRE_SAME_NDIM_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			);

		attr_.device_dispatcher_forward = layer_attribute::OUTPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::OUTPUT_BASE;
	}

	concat_layer::~concat_layer() {
		release_all();
	}

	void concat_layer::init() {
		check_input();
		concat_dim_ = inputs_[0]->get(0)->ndim() - 1 - (int)layer_param_.concat_param().concat_dim();
		CHECK_GE(concat_dim_, 0);

		CHECK(concat_dim_ == inputs_[0]->get(0)->ndim() - 1 || concat_dim_ == inputs_[0]->get(0)->ndim_inner() - 1)
			<< "concat_dim along dim" << concat_dim_ << " not implemented yet";
	}

	void concat_layer::resize() {
		check_input();
		int n_inputs = (int)inputs_.size();
		int n_devices = (int)inputs_[0]->size();
		int n_dim = (int)inputs_[0]->get(0)->ndim();
		int n_dim_inner = (int)inputs_[0]->get(0)->ndim_inner();
		auto dim_type = inputs_[0]->get(0)->dim_type();
		bool init = (outputs_[0]->size() == 0);

		if (!init) {
			bool reshaped = false;
			for (int i = 0; i < n_inputs; i++) {
				if (inputs_[i]->reshaped()) {
					reshaped = true;
					break;
				}
			}
			if (!reshaped) {
				return;
			}
		}

		CHECK_LT(concat_dim_, n_dim);

		if (dim_type == device_blob::DIMTYPE_FIXED_LEN) {
			for (int nd = 0; nd < n_devices; nd++) {
				vector<int> dims = inputs_[0]->get(nd)->dims();

				for (int i = 1; i < n_inputs; i++) {
					const vector<int> &cur_dims = inputs_[i]->get(nd)->dims();
					for (int j = 0; j < n_dim; j++) {
						if (j != concat_dim_) {
							CHECK_EQ(dims[j], cur_dims[j]);
						}
						else {
							dims[j] += cur_dims[j];
						}
					}
				}

				if (init) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(
						device_blob::create_nd(context_, n_dim, &dims[0], inputs_[0]->get(nd)->device_id())
						));
				}
				else {
					outputs_[0]->get(nd)->reshape_nd(n_dim, &dims[0]);
				}
			}
		}
		else if (dim_type == device_blob::DIMTYPE_EXT_LEN) {
			for (int nd = 0; nd < n_devices; nd++) {
				int new_num = inputs_[0]->get(nd)->num();
				int dim_size = n_dim_inner * inputs_[0]->get(nd)->num();
				vector<int> ext_dims(dim_size);
				memcpy(&ext_dims[0], inputs_[0]->get(nd)->ext_dims_cpu(), dim_size * sizeof(int));

				for (int i = 1; i < n_inputs; i++) {
					const int *cur_ext_dims = inputs_[i]->get(nd)->ext_dims_cpu();
					int cur_num = (int)inputs_[i]->get(nd)->num();

					if (concat_dim_ < n_dim - 1) { // concat inner dims
						CHECK_EQ(cur_num, new_num);

						for (int n = 0; n < cur_num; n++) {
							for (int j = 0; j < n_dim_inner; j++) {
								int index = n * n_dim_inner + j;
								if (j != concat_dim_) {
									CHECK_EQ(ext_dims[index], cur_ext_dims[index]);
								}
								else {
									ext_dims[index] += cur_ext_dims[index];
								}
							}
						}
					}
					else { // concat num
						new_num += cur_num;

						int cur_dim_size = n_dim_inner * cur_num;
						ext_dims.insert(ext_dims.end(), cur_ext_dims, cur_ext_dims + cur_dim_size);
					}
				}

				if (init) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(
						device_blob::create_ext(context_, n_dim_inner, new_num, &ext_dims[0], inputs_[0]->get(nd)->device_id())
						));
				}
				else {
					outputs_[0]->get(nd)->reshape_ext(n_dim_inner, new_num, &ext_dims[0]);
				}
			}
		}
	}

	void concat_layer::on_forward(int device_index) {
		int n_dim = inputs_[0]->get(device_index)->ndim();
		int n_dim_inner = inputs_[0]->get(device_index)->ndim_inner();
		
		if (concat_dim_ == n_dim - 1) { // concat num
			int offset = 0;
			for (int i = 0; i < (int)inputs_.size(); i++) {
				CUDA_CHECK(cudaMemcpyAsync(
					outputs_[0]->get(device_index)->mutable_gpu_data() + offset,
					inputs_[i]->get(device_index)->gpu_data(),
					inputs_[i]->get(device_index)->count() * sizeof(data_type),
					cudaMemcpyDeviceToDevice
					));
				offset += inputs_[0]->get(device_index)->count();
			}
		}
		else {
			CHECK_EQ(concat_dim_, n_dim_inner - 1);

			if (inputs_[0]->get(device_index)->dim_type() == device_blob::DIMTYPE_FIXED_LEN) {
				int line_stride = outputs_[0]->get(device_index)->inner_count();
				int offset = 0;
				for (int i = 0; i < (int)inputs_.size(); i++) {
					CUDA_CHECK(cudaMemcpy2DAsync(
						outputs_[0]->get(device_index)->mutable_gpu_data() + offset,
						line_stride * sizeof(data_type),
						inputs_[i]->get(device_index)->gpu_data(),
						inputs_[i]->get(device_index)->inner_count() * sizeof(data_type),
						inputs_[i]->get(device_index)->inner_count() * sizeof(data_type),
						inputs_[i]->get(device_index)->num(),
						cudaMemcpyDeviceToDevice
						));
					offset += inputs_[i]->get(device_index)->inner_count();
				}
			}
			else if (inputs_[0]->get(device_index)->dim_type() == device_blob::DIMTYPE_EXT_LEN) {
				vector<int> offset_channel(inputs_[0]->get(device_index)->num());
				memset(&offset_channel[0], 0, offset_channel.size() * sizeof(int));
				for (int i = 0; i < (int)inputs_.size(); i++) {
					inputs_[i]->get(device_index)->sync_ext_dim();
					const int *ext_dims = inputs_[i]->get(device_index)->ext_dims_cpu();
					for (int n = 0; n < (int)inputs_[i]->get(device_index)->num(); n++) {
						const data_type *bottom_data = inputs_[i]->get(device_index)->gpu_data()
							+ inputs_[i]->get(device_index)->offset(n);
						data_type *top_data = outputs_[0]->get(device_index)->mutable_gpu_data()
							+ outputs_[0]->get(device_index)->offset(n) + offset_channel[n];

						int count = 1;
						const int *cur_ext_dims = ext_dims + n * n_dim_inner;
						for (int j = 0; j < n_dim_inner; j++) {
							count *= cur_ext_dims[j];
						}

						CUDA_CHECK(cudaMemcpyAsync(
							top_data,
							bottom_data,
							count * sizeof(data_type),
							cudaMemcpyDeviceToDevice
							));

						offset_channel[n] += count;
					}
				}
			}
		}
	}

	void concat_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		int n_dim = inputs_[0]->get(device_index)->ndim();
		int n_dim_inner = inputs_[0]->get(device_index)->ndim_inner();
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		if (concat_dim_ == n_dim - 1) { // concat num
			int offset = 0;
			for (int i = 0; i < (int)inputs_.size(); i++) {
				if (should_bp(bp_acts, i)) {
					const data_type beta = get_beta(clear_acts_diff, i);

					if (beta == 0) {
						CUDA_CHECK(cudaMemcpyAsync(
							inputs_[i]->get(device_index)->mutable_gpu_diff(),
							outputs_[0]->get(device_index)->gpu_diff() + offset,
							inputs_[i]->get(device_index)->count() * sizeof(data_type),
							cudaMemcpyDeviceToDevice
							));
					}
					else {
						cublas.axpby(
							inputs_[i]->get(device_index)->count(),
							(data_type)1.f,
							outputs_[0]->get(device_index)->gpu_diff() + offset,
							beta,
							inputs_[i]->get(device_index)->mutable_gpu_diff()
							);
					}
				}
				offset += inputs_[0]->get(device_index)->count();
			}
		}
		else { // concat channel
			CHECK_EQ(concat_dim_, n_dim_inner - 1);

			if (inputs_[0]->get(device_index)->dim_type() == device_blob::DIMTYPE_FIXED_LEN) {

				int line_stride = outputs_[0]->get(device_index)->inner_count();
				int offset = 0;

				for (int i = 0; i < (int)inputs_.size(); i++) {
					if (should_bp(bp_acts, i)) {
						const data_type beta = get_beta(clear_acts_diff, i);

						if (beta == 0) {
							CUDA_CHECK(cudaMemcpy2DAsync(
								inputs_[i]->get(device_index)->mutable_gpu_diff(),
								inputs_[i]->get(device_index)->inner_count() * sizeof(data_type),
								outputs_[0]->get(device_index)->gpu_diff() + offset,
								line_stride * sizeof(data_type),
								inputs_[i]->get(device_index)->inner_count() * sizeof(data_type),
								inputs_[i]->get(device_index)->num(),
								cudaMemcpyDeviceToDevice
								));
						}
						else {
							int count = inputs_[i]->get(device_index)->count();
							void *buffer = context_->get_current_device()->memory()->allocate(count * sizeof(data_type));
							CUDA_CHECK(cudaMemcpy2DAsync(
								buffer,
								inputs_[i]->get(device_index)->inner_count() * sizeof(data_type),
								outputs_[0]->get(device_index)->gpu_diff() + offset,
								line_stride * sizeof(data_type),
								inputs_[i]->get(device_index)->inner_count() * sizeof(data_type),
								inputs_[i]->get(device_index)->num(),
								cudaMemcpyDeviceToDevice
								));
							cublas.axpby(
								count,
								(data_type)1.f,
								reinterpret_cast<data_type *>(buffer),
								beta,
								inputs_[i]->get(device_index)->mutable_gpu_diff()
								);

							context_->get_current_device()->memory()->free(buffer);
						}
					}

					offset += inputs_[i]->get(device_index)->inner_count();
				}
			}
			else if (inputs_[0]->get(device_index)->dim_type() == device_blob::DIMTYPE_EXT_LEN) {
				vector<int> offset_channel(inputs_[0]->get(device_index)->num());
				memset(&offset_channel[0], 0, offset_channel.size() * sizeof(int));
				for (int i = 0; i < (int)inputs_.size(); i++) {
					inputs_[i]->get(device_index)->sync_ext_dim();
					const int *ext_dims = inputs_[i]->get(device_index)->ext_dims_cpu();
					for (int n = 0; n < (int)inputs_[i]->get(device_index)->num(); n++) {
						data_type *bottom_diff = inputs_[i]->get(device_index)->mutable_gpu_diff()
							+ inputs_[i]->get(device_index)->offset(n);
						const data_type *top_diff = outputs_[0]->get(device_index)->gpu_diff()
							+ outputs_[0]->get(device_index)->offset(n) + offset_channel[n];

						int count = 1;
						const int *cur_ext_dims = ext_dims + n * n_dim_inner;
						for (int j = 0; j < n_dim_inner; j++) {
							count *= cur_ext_dims[j];
						}

						if (should_bp(bp_acts, i)) {
							const data_type beta = get_beta(clear_acts_diff, i);

							if (beta == 0) {
								CUDA_CHECK(cudaMemcpyAsync(
									bottom_diff,
									top_diff,
									count * sizeof(data_type),
									cudaMemcpyDeviceToDevice
									));
							}
							else {
								cublas.axpby(
									count,
									(data_type)1.f,
									top_diff,
									beta,
									bottom_diff
									);
							}
						}

						offset_channel[n] += count;
					}
				}
			}
		}
	}
}