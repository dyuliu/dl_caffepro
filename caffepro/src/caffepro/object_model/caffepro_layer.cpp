
#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	caffepro_layer::caffepro_layer(caffepro_context *context, const LayerParameter &param) 
		: context_(context), layer_param_(param) {
		memset(&attr_, 0, sizeof(layer_attribute));
	}

	caffepro_layer::~caffepro_layer() {
		// nothing to do
	}

	void caffepro_layer::bind(layer_io_buffer &inputs, layer_io_buffer &outputs) {
		CHECK_GE(inputs.size(), attr_.num_inputs_min);
		CHECK_LE(inputs.size(), attr_.num_inputs_max);
		CHECK_GE(outputs.size(), attr_.num_outputs_min);
		CHECK_LE(outputs.size(), attr_.num_outputs_max);
	
		for (int i = 0; i < (int)inputs.size(); i++) {
			CHECK(inputs[i]) << "input buffer uninitialized";
		}

		for (int i = 0; i < (int)outputs.size(); i++) {
			CHECK(outputs[i]) << "output buffer uninitialized";
		}

		// copy buffer
		inputs_ = inputs;
		outputs_ = outputs;
		
		if (attr_.get_constraint(layer_attribute::CF_FORBID_INPLACE_USAGE_PREV_ALWAYS) && inputs_.size() > 0) {
			CHECK(!inputs_[0]->get_attr(node_blob::NF_BIND_INPLACE));
		}
		
		if (inplace()) {
			CHECK(attr_.get_constraint(layer_attribute::CF_ALLOW_INPLACE)) << "inplace calculation does not supported";

			CHECK(!attr_.get_constraint(layer_attribute::CF_FORBID_INPLACE_USAGE_PREV_WHEN_INPLACE) 
				|| !inputs_[0]->get_attr(node_blob::NF_BIND_INPLACE)) << "incompatible with inplace input";

			CHECK(!outputs_[0]->get_attr(node_blob::NF_BIND_FORBID_INPLACE_USAGE)) << "previous layer requires not using inplace calculation";
			
			if (attr_.get_constraint(layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_WHEN_INPLACE)) {
				outputs_[0]->set_attr(node_blob::NF_BIND_FORBID_INPLACE_USAGE);
			}

			outputs_[0]->set_attr(node_blob::NF_BIND_INPLACE);
		}

		if (attr_.get_constraint(layer_attribute::CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS) && outputs_.size() > 0) {
			outputs_[0]->set_attr(node_blob::NF_BIND_FORBID_INPLACE_USAGE);
		}

		for (int i = 0; i < (int)inputs_.size(); i++) {
			inputs_[i]->bind_as_layer_input(this);
		}

		for (int i = 0; i < (int)outputs_.size(); i++) {
			outputs_[i]->bind_as_layer_output(this);
		}
	}

	void caffepro_layer::check_input() {
		// check input requirement
		for (int i = 0; i < (int)inputs_.size(); i++) {
			if (!inputs_[i]->reshaped()) continue;

			CHECK_GT(inputs_[i]->size(), 0) << "input buffer has not set up";

			if (attr_.get_constraint(layer_attribute::CF_REQUIRE_UNIQUE_DEVICE)) {
				CHECK_EQ(inputs_[i]->size(), 1) << "unique device required";
			}

			for (int nd = 0; nd < (int)inputs_[i]->size(); nd++) {
				CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_4D)
					|| inputs_[i]->get(nd)->is_4d()) << "4D blob required";

				CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_FIXEDLEN_DIM)
					|| inputs_[i]->get(nd)->dim_type() == device_blob::DIMTYPE_FIXED_LEN) << "fixed-len blob required";

				CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_NDIM_4)
					|| inputs_[i]->get(nd)->ndim() == 4) << "NDIM should be 4";

				CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_INNER_COUNT)
					|| inputs_[i]->get(nd)->get_attr(device_blob::BF_DIM_SAME_COUNT));
			}

			CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES)
				|| inputs_[i]->same_dimtype());
			CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_NDIM_ACROSS_DEVICES)
				|| inputs_[i]->same_ndim());
			CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_COUNT_ACROSS_DEVICES)
				|| inputs_[i]->same_count());
			CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_NUM_ACROSS_DEVICES)
				|| inputs_[i]->same_num());
			CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_INNER_COUNT_ACROSS_DEVICES)
				|| inputs_[i]->same_inner_count());

			if (i > 0) {
				if (attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_DEVICE)) {
					CHECK_EQ(inputs_[i]->size(), inputs_[0]->size());
				}

				if (inputs_[i]->size() == inputs_[0]->size()) {
					for (int nd = 0; nd < (int)inputs_[i]->size(); nd++) {
						if (attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_DEVICE)) {
							CHECK_EQ(inputs_[i]->get(nd)->device_id(), inputs_[0]->get(nd)->device_id());
						}

						CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_SHAPE) 
							|| inputs_[i]->get(nd)->same_shape(*inputs_[0]->get(nd)));
						CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_COUNT)
							|| inputs_[i]->get(nd)->count() == inputs_[0]->get(nd)->count());
						CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_NUM)
							|| inputs_[i]->get(nd)->num() == inputs_[0]->get(nd)->num());
						CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_DIMTYPE)
							|| inputs_[i]->get(nd)->dim_type() == inputs_[0]->get(nd)->dim_type());
						CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_NDIM)
							|| inputs_[i]->get(nd)->ndim() == inputs_[0]->get(nd)->ndim());
						CHECK(!attr_.get_constraint(layer_attribute::CF_REQUIRE_SAME_INNER_COUNT)
							|| inputs_[i]->get(nd)->inner_count() == inputs_[0]->get(nd)->inner_count());
					}
				}
			}
		}
	}

	void caffepro_layer::init() {
		check_input();
	}

	void caffepro_layer::resize() {
		check_input();

		// by default, create outputs same with the first input

		if (!inplace()) {
			CHECK_GT(inputs_.size(), 0);

			for (int i = 0; i < (int)outputs_.size(); i++) {
				if (outputs_[i]->size() == 0) {	// init
					outputs_[i]->add_like(*inputs_[0]);
				}
				else {	// reshape
					CHECK_EQ(outputs_[i]->size(), inputs_[0]->size());
					for (int nd = 0; nd < (int)outputs_[i]->size(); nd++) {
						if (inputs_[0]->get(nd)->reshaped()) {
							if (!outputs_[i]->get(nd)->same_shape(*inputs_[0]->get(nd))) {
								outputs_[i]->get(nd)->reshape_like(*inputs_[0]->get(nd));
							}
						}
					}
				}
			}
		}
	}

	void caffepro_layer::forward() {
		on_before_forward();

		if (attr_.device_dispatcher_forward == layer_attribute::INPUT_BASE) {
			CHECK_GT(inputs_.size(), 0);
			CHECK_GT(inputs_[0]->size(), 0);

			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				ENTER_DEVICE_CONTEXT(inputs_[0]->get(nd)->device_id())
					on_forward(nd);
				EXIT_DEVICE_CONTEXT;
			}
		}
		else if (attr_.device_dispatcher_forward == layer_attribute::OUTPUT_BASE) {
			CHECK_GT(outputs_.size(), 0);
			CHECK_GT(outputs_[0]->size(), 0);

			for (int nd = 0; nd < (int)outputs_[0]->size(); nd++) {
				ENTER_DEVICE_CONTEXT(outputs_[0]->get(nd)->device_id())
					on_forward(nd);
				EXIT_DEVICE_CONTEXT;
			}
		}

		on_after_forward();
	}

	void caffepro_layer::backward(act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		on_before_backward();

		if (attr_.device_dispatcher_backward == layer_attribute::INPUT_BASE) {
			CHECK_GT(inputs_.size(), 0);
			CHECK_GT(inputs_[0]->size(), 0);

			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				ENTER_DEVICE_CONTEXT(inputs_[0]->get(nd)->device_id())
					on_backward(nd, bp_acts, bp_weights, clear_acts_diff, clear_weights_diff);
				EXIT_DEVICE_CONTEXT;
			}
		}
		else if (attr_.device_dispatcher_backward == layer_attribute::OUTPUT_BASE) {
			CHECK_GT(outputs_.size(), 0);
			CHECK_GT(outputs_[0]->size(), 0);

			for (int nd = 0; nd < (int)outputs_[0]->size(); nd++) {
				ENTER_DEVICE_CONTEXT(outputs_[0]->get(nd)->device_id())
					on_backward(nd, bp_acts, bp_weights, clear_acts_diff, clear_weights_diff);
				EXIT_DEVICE_CONTEXT;
			}
		}

		on_after_backward();
	}

	void caffepro_layer::release_all() {
		// nothing to do
	}

	void caffepro_layer::write_to_proto(LayerParameter *proto) {
		proto->Clear();
		proto->CopyFrom(layer_param_);
		proto->clear_blobs();
		for (int i = 0; i < weights_.size(); ++i) {
			CHECK_GT(weights_[i]->size(), 0);
			weights_[i]->save_data_to(proto->add_blobs());
		}
	}
}