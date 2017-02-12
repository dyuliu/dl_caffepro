
#include <caffepro/layers/gpu_concat_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	gpu_concat_layer::gpu_concat_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;
		attr_.device_dispatcher_forward = layer_attribute::INPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::INPUT_BASE;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_NDIM_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			);
	}

	gpu_concat_layer::~gpu_concat_layer() {
		release_all();
	}

	void gpu_concat_layer::init() {
		check_input();

		concated_device_id_ = layer_param_.gpu_id();
		CHECK_GE(concated_device_id_, 0);
	}

	void gpu_concat_layer::resize() {
		check_input();

		if (outputs_[0]->size() == 0 || inputs_[0]->reshaped()) {

			nd_offsets_.resize(inputs_[0]->size());
			nd_offsets_[0] = 0;
			for (int nd = 1; nd < (int)inputs_[0]->size(); nd++) {
				nd_offsets_[nd] = nd_offsets_[nd - 1] + inputs_[0]->get(nd - 1)->num();
			}

			if (inputs_[0]->get(0)->dim_type() == device_blob::DIMTYPE_FIXED_LEN) {
				int ndim_inner = inputs_[0]->get(0)->ndim_inner();
				for (int i = 0; i < ndim_inner; i++) {
					CHECK(inputs_[0]->same_dim_at(i));
				}

				int sum_num = inputs_[0]->sum_num();
				std::vector<int> dims = inputs_[0]->get(0)->dims();
				dims.back() = sum_num;

				if (outputs_[0]->size() == 0) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(
						device_blob::create_nd(context_, (int)dims.size(), &dims[0], concated_device_id_)
						));
				}
				else {
					outputs_[0]->get(0)->reshape_nd((int)dims.size(), &dims[0]);
				}
			}
			else if (inputs_[0]->get(0)->dim_type() == device_blob::DIMTYPE_EXT_LEN) {
				int ndim_inner = inputs_[0]->get(0)->ndim_inner();
				int sum_num = inputs_[0]->sum_num();
				std::vector<int> ext_dims(ndim_inner * sum_num);

				int offset = 0;
				for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
					const int *dim_data = inputs_[0]->get(nd)->ext_dims_cpu();
					int n = inputs_[0]->get(nd)->num() * ndim_inner;
					memcpy(&ext_dims[offset], dim_data, n * sizeof(int));
					
					offset += n;
				}

				if (outputs_[0]->size() == 0) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(
						device_blob::create_ext(context_, ndim_inner, sum_num, &ext_dims[0], concated_device_id_)
						));
				}
				else {
					outputs_[0]->get(0)->reshape_ext(ndim_inner, sum_num, &ext_dims[0]);
				}
			}
		}
	}

	void gpu_concat_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(0)->mutable_gpu_data() + outputs_[0]->get(0)->offset(nd_offsets_[device_index]);

		CUDA_CHECK(cudaMemcpyAsync(top_data, bottom_data, 
			inputs_[0]->get(device_index)->count() * sizeof(data_type), cudaMemcpyDeviceToDevice));
	}

	void gpu_concat_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type alpha = 0;
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			const data_type* top_diff = outputs_[0]->get(0)->gpu_diff()
				+ outputs_[0]->get(0)->offset(nd_offsets_[device_index]);
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();
			int count = inputs_[0]->get(device_index)->count();

			if (beta_acts == 0) {
				CUDA_CHECK(cudaMemcpyAsync(bottom_diff, top_diff,
					count * sizeof(data_type), cudaMemcpyDeviceToDevice));
			}
			else {
				void *buffer = context_->get_current_device()->memory()->allocate(count * sizeof(data_type));
				CUDA_CHECK(cudaMemcpyAsync(buffer, top_diff,
					count * sizeof(data_type), cudaMemcpyDeviceToDevice));
				cublas_wrapper<data_type> cublas(context_, context()->get_current_device()->device_id());
				cublas.axpby(count, alpha, reinterpret_cast<const data_type *>(buffer), beta_acts, bottom_diff);
				context()->get_current_device()->memory()->free(buffer);
			}
		}
	}

	void gpu_concat_layer::on_after_forward() {
		context_->sync_all_devices();
	}

	void gpu_concat_layer::on_before_backward() {
		context_->sync_all_devices();
	}

	void gpu_concat_layer::on_after_backward() {
		context_->sync_all_devices();
	}
}