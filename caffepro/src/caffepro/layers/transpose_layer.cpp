
#include <caffepro/layers/transpose_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	transpose_layer::transpose_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	transpose_layer::~transpose_layer() {
		release_all();
	}

	void transpose_layer::init() {
		check_input();

		lead_dim_ = layer_param_.transpose_param().lead_dim();
		CHECK_GT(lead_dim_, 0); // lead_dim_ == 0 means doing nothing. Not allowed here

		auto &dims = layer_param_.transpose_param().output_dims();

		bool has_zero = false;
		for (int i = 0; i < (int)dims.size(); i++) {
			output_dims_.push_back(dims.Get(i));
			CHECK_GE(output_dims_.back(), 0);
			if (output_dims_.back() == 0) {
				CHECK(!has_zero);
				has_zero = true;
			}
		}
	}

	void transpose_layer::resize() {
		check_input();

		bool init = (outputs_[0]->size() == 0);
		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			auto &input = *inputs_[0]->get(nd);
			
			if (input.reshaped()) {
				std::vector<int> dims;
				CHECK_LT(lead_dim_, input.ndim());

				if (output_dims_.size() == 0) {
					for (int i = lead_dim_; i < input.ndim(); i++) {
						dims.push_back(input.dim_at(i));
					}
					for (int i = 0; i < lead_dim_; i++) {
						dims.push_back(input.dim_at(i));
					}
				}
				else {
					dims = output_dims_;
					int dim_to_fill = -1, count = 1;
					for (int i = 0; i < (int)dims.size(); i++) {
						if (dims[i] == 0) {
							dim_to_fill = i;
						}
						else {
							count *= dims[i];
						}
					}

					if (dim_to_fill >= 0) {
						CHECK_EQ(input.count() % count, 0);
						dims[dim_to_fill] = input.count() / count;
					}
					else {
						CHECK_EQ(input.count(), count);
					}
				}

				if (init) {
					outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_nd(
						context_, (int)dims.size(), &dims[0], input.device_id()
						)));
				}
				else {
					outputs_[0]->get(nd)->reshape_nd((int)dims.size(), &dims[0]);
				}
			}
		}
	}

	void transpose_layer::on_forward(int device_index) {
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		auto &input = *inputs_[0]->get(device_index);

		int r = 1, c = 1;
		for (int i = lead_dim_; i < input.ndim(); i++) {
			r *= input.dim_at(i);
		}
		for (int i = 0; i < lead_dim_; i++) {
			c *= input.dim_at(i);
		}

		cublas.transpose(r, c, input.gpu_data(), outputs_[0]->get(device_index)->mutable_gpu_data());
	}

	void transpose_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0)) {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			auto &input = *inputs_[0]->get(device_index);

			int r = 1, c = 1;
			for (int i = lead_dim_; i < input.ndim(); i++) {
				r *= input.dim_at(i);
			}
			for (int i = 0; i < lead_dim_; i++) {
				c *= input.dim_at(i);
			}

			data_type beta = get_beta(clear_acts_diff, 0);
			cublas.transpose_add(c, r, outputs_[0]->get(device_index)->gpu_diff(), beta, input.mutable_gpu_diff());
		}
	}
}