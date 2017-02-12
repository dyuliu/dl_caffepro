
#include <caffepro/layers/data_entry_layer.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/context/common_names.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	using std::string;

	data_entry_layer::data_entry_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		
		attr_.num_inputs_min = attr_.num_inputs_max = 0;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.device_dispatcher_forward = layer_attribute::OUTPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::OUTPUT_BASE;
		attr_.usage = layer_attribute::USAGE_DATA_SOURCE;
	}

	data_entry_layer::~data_entry_layer() {
		// nothing to do
	}

	void data_entry_layer::init() {
		caffepro_object *data_source = context_->get_shared_object(namespace_, SHAREDOBJNAME_DATASOURCE);
		data_model::data_provider *provider = dynamic_cast<data_model::data_provider *>(data_source);
		CHECK(provider) << "Data provider is required by data entry layer";

		CHECK(provider->get_entry(layer_param_.data_entry_param().entry_name()))
			<< "Missing data entry: " << layer_param_.data_entry_param().entry_name();

		data_model::data_entry &entry = *provider->get_entry(layer_param_.data_entry_param().entry_name());
		int entry_index = layer_param_.data_entry_param().entry_index();
		CHECK_LT(entry_index, entry.outputs().size()) << "Entry index out of range";

		CHECK(!entry.outputs()[entry_index] || entry.outputs()[entry_index]->output_bindings().size() == 0)
			<< "Multiple data entry layers sharing the same data entry output is not allowed";

		outputs_[0]->set_attr(node_blob::NF_NET_INPUT);
		entry.outputs()[entry_index] = outputs_[0];
		entry.set_enabled(true);
		entry.forward();
	}

	void data_entry_layer::resize() {
		// nothing to do
	}

	void data_entry_layer::on_forward(int device_index) {
		// nothing to do
	}

	void data_entry_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		// nothing to do
	}
}