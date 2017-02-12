
#include <caffepro/layers/weight_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <boost/scoped_ptr.hpp>

namespace caffepro {
	using std::vector;
	
	weight_layer::weight_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = 0;
		attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.device_dispatcher_forward = layer_attribute::OUTPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::OUTPUT_BASE;
	}

	weight_layer::~weight_layer() {
		// do nothing
	}

	void weight_layer::init() {
		check_input();

		vector<int> dims;
		for (int i = 0; i < (int)layer_param_.weight_param().weight_dim_size(); i++) {
			dims.push_back((int)layer_param_.weight_param().weight_dim(i));
			CHECK_GT(dims.back(), 0);
		}

		CHECK_GT(dims.size(), 0);

		// add weights
		weights_.push_back(boost::shared_ptr<node_blob>(new node_blob()));
		if (inputs_.size() == 0) {
			// create weights on default device
			weights_[0]->add(boost::shared_ptr<device_blob>(
				device_blob::create_nd(context_, (int)dims.size(), &dims[0], -1)
			));
		}
		else {
			// create weights based on input devices
			boost::scoped_ptr<device_blob> weight_template(
				device_blob::create_nd(context_, (int)dims.size(), &dims[0], -1));
			weights_[0]->add_like(context_, *weight_template, *inputs_[0]);
		}

		// init weights
		boost::scoped_ptr<filler> weight_filler(filler::create(context_, layer_param_.weight_param().weight_filler()));
		weight_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		// init outputs
		outputs_[0]->add_like(*weights_[0]);
	}

	void weight_layer::resize() {
		check_input();
	}

	void weight_layer::on_forward(int device_index) {
		outputs_[0]->get(device_index)->copy_data_from_via_gpu(*weights_[0]->get(device_index));
	}

	void weight_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (inputs_.size() > 0) {
			if (should_bp(bp_acts, 0) && get_beta(clear_acts_diff, 0) == 0) {
				inputs_[0]->get(device_index)->fill_diff(0.f);
			}
		}

		if (should_bp(bp_weights, 0)) {
			data_type beta = get_beta(clear_weights_diff, 0);
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.axpby(
				weights_[0]->get(device_index)->count(),
				1.f,
				outputs_[0]->get(device_index)->gpu_diff(),
				beta,
				weights_[0]->get(device_index)->mutable_gpu_diff()
				);
		}
	}
}