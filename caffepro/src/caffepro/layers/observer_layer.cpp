
#include <caffepro/layers/observer_layer.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	observer_layer::observer_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;
	}

	observer_layer::~observer_layer() {
		release_all();
	}

	void observer_layer::init() {
		check_input();

		int n_devices = (int)inputs_[0]->size();

		for (int nd = 0; nd < n_devices; nd++) {
			outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
				context_, 1, 4, 1, 1, inputs_[0]->get(nd)->device_id()
				)));
		}
		outputs_[0]->tags().resize(4);
		outputs_[0]->tags()[0] = "Mean";
		outputs_[0]->tags()[1] = "Std";
		outputs_[0]->tags()[2] = "Max";
		outputs_[0]->tags()[3] = "Min";
	}

	void observer_layer::resize() {
		check_input();

		// nothing to do
	}

	void observer_layer::on_forward(int device_index) {
		data_type *stat = outputs_[0]->get(device_index)->mutable_cpu_data();
		stat[0] = (data_type)inputs_[0]->get(device_index)->mean();
		stat[1] = (data_type)sqrt(inputs_[0]->get(device_index)->variance());
		stat[2] = inputs_[0]->get(device_index)->max();
		stat[3] = inputs_[0]->get(device_index)->min();
	}

	void observer_layer::on_after_forward() {
		int n_devices = (int)outputs_[0]->size();
		int count = outputs_[0]->get(0)->count();
		data_type *stat = outputs_[0]->get(0)->mutable_cpu_data();

		for (int nd = 1; nd < n_devices; nd++) {
			const data_type *dev_stat = outputs_[0]->get(nd)->cpu_data();
			for (int i = 0; i < count; i++) {
				stat[i] += dev_stat[i];
			}
		}

		for (int i = 0; i < count; i++) {
			stat[i] /= n_devices;
		}
	}

	void observer_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		if (should_bp(bp_acts, 0) && get_beta(clear_acts_diff, 0) == 0) {
			inputs_[0]->get(device_index)->fill_diff(0.f);
		}
	}
}