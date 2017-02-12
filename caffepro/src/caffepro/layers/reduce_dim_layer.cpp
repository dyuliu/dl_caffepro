
#include <caffepro/layers/reduce_dim_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	reduce_dim_layer::reduce_dim_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	reduce_dim_layer::~reduce_dim_layer() {
		release_all();
	}

	void reduce_dim_layer::init() {
		check_input();

		dim_ = layer_param_.reduce_dim_param().dim();
		group_ = layer_param_.reduce_dim_param().group();
		sum_multiplier_.reset(new node_blob());
	}

	void reduce_dim_layer::resize() {
		check_input();

		int n_devices = (int)inputs_[0]->size();
		bool init = (outputs_[0]->size() == 0);
		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped()) {
				auto &input = *inputs_[0]->get(nd);
				CHECK_LT(dim_, input.ndim());
				
				int original_dim = input.dim_at(dim_);
				CHECK_EQ(original_dim % group_, 0);

				std::vector<int> dims = input.dims();
				dims[dim_] = group_;

				int reduce_len = original_dim / group_;

				if (init) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_nd(
						context_, input.ndim(), &dims[0], input.device_id()
						)));
				}
				else {
					outputs_[0]->get(nd)->reshape_nd(input.ndim(), &dims[0]);
				}

				sum_multiplier_->set_4d(nd, reduce_len, 1, 1, 1, input.device_id(), context_);
				sum_multiplier_->get(nd)->fill_data(1.f);
			}
		}
	}

	void reduce_dim_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);
		auto &output = *outputs_[0]->get(device_index);
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		int original_dim = input.dim_at(dim_);
		int reduce_len = original_dim / group_;

		data_type alpha = 0.f;
		if (layer_param_.reduce_dim_param().reduce_type() == layer_param_.reduce_dim_param().AVE) {
			alpha = (data_type)1.f / reduce_len;
		}
		else if (layer_param_.reduce_dim_param().reduce_type() == layer_param_.reduce_dim_param().SUM) {
			alpha = 1.f;
		}
		else {
			LOG(FATAL) << "Unknown reduce type" << layer_param_.reduce_dim_param().reduce_type();
		}

		int ndim = input.ndim();
		if (dim_ == ndim - 1) {
			cublas.gemv(CblasTrans, reduce_len, input.count() / reduce_len, alpha,
				input.gpu_data(), sum_multiplier_->get(device_index)->gpu_data(), (data_type)0.f, output.mutable_gpu_data());
		}
		else {
			data_type *buffer = reinterpret_cast<data_type *>(context_->get_current_device()->memory()->allocate(input.count() * sizeof(data_type)));
			int high_end = 1;
			for (int i = dim_ + 1; i < ndim; i++) {
				high_end *= input.dim_at(i);
			}
			cublas.transpose(high_end, input.count() / high_end, input.gpu_data(), buffer);
			cublas.gemv(CblasTrans, reduce_len, input.count() / reduce_len, alpha,
				buffer, sum_multiplier_->get(device_index)->gpu_data(), (data_type)0.f, output.mutable_gpu_data());
			cublas.transpose(output.count() / high_end, high_end, output.gpu_data(), buffer);
			cublas.copy(output.count(), buffer, output.mutable_gpu_data());
			context_->get_current_device()->memory()->free(buffer);
		}
	}

	void reduce_dim_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		NOT_IMPLEMENTED;
	}
}