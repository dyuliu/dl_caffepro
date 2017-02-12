
#include <caffepro/layers/correlation_layer.h>
#include <caffepro/layers/conv_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>

namespace caffepro {
	correlation_layer::correlation_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	correlation_layer::~correlation_layer() {
		release_all();
	}

	void correlation_layer::init() {
		check_input();

		// build conv param
		conv_parameter_.reset(new LayerParameter());
		ConvolutionParameter &conv_param = *conv_parameter_->mutable_conv_param();
		const CorrelationParameter &src_param = layer_param_.correlation_param();
		
		if (src_param.block_calc_size() > 0) {
			conv_param.add_block_calc(src_param.block_calc(0));
		}
		conv_param.set_kernel_size_x(inputs_[1]->get(0)->width());
		conv_param.set_kernel_size_y(inputs_[1]->get(0)->height());
		if (src_param.has_stride_x() || src_param.has_stride_y()) {
			conv_param.set_stride_x(src_param.stride_x());
			conv_param.set_stride_y(src_param.stride_y());
		}
		else {
			conv_param.set_stride(src_param.stride());
		}
		if (src_param.has_pad_x() || src_param.has_pad_y()) {
			conv_param.set_pad_x(src_param.pad_x());
			conv_param.set_pad_y(src_param.pad_y());
		}
		else {
			conv_param.set_pad(src_param.pad());
		}
		conv_param.set_bias_term(src_param.bias_term());
		conv_param.set_size_floor(src_param.size_floor());
		conv_param.set_num_output(inputs_[1]->get(0)->num());
		*conv_param.mutable_bias_filler() = src_param.bias_filler();

		CHECK_EQ(inputs_[0]->get(0)->channels(), inputs_[1]->get(0)->channels());
		channels_ = inputs_[0]->get(0)->channels();

		// build conv layer
		conv_.reset(new conv_layer(context_, *conv_parameter_));

		// prepare i/o binding
		conv_inputs_.push_back(inputs_[0]);
		conv_outputs_.push_back(outputs_[0]);
		conv_->bind(conv_inputs_, conv_outputs_);

		// init conv layer
		conv_->init();

		// set up bias
		if (src_param.bias_term()) {
			weights_.resize(1);
			weights_[0] = conv_->weights()[1];
		}

		// set up weights
		conv_->weights()[0] = inputs_[1];
	}

	void correlation_layer::resize() {
		check_input();

		int n_device = (int)inputs_[1]->size();
		for (int nd = 0; nd < n_device; nd++) {
			if (inputs_[1]->get(nd)->reshaped()) {
				CHECK_EQ(inputs_[1]->get(nd)->width(), conv_parameter_->conv_param().kernel_size_x());
				CHECK_EQ(inputs_[1]->get(nd)->height(), conv_parameter_->conv_param().kernel_size_y());
				CHECK_EQ(inputs_[1]->get(nd)->channels(), channels_);
				CHECK_EQ(inputs_[1]->get(nd)->num(), conv_parameter_->conv_param().num_output());
			}
		}

		conv_->resize();
	}

	void correlation_layer::forward() {
		conv_->forward();
	}

	void correlation_layer::backward(act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		act_selector conv_act_selector = bp_acts & 1;
		weight_selector conv_weight_selector = ((bp_weights & 1) << 1) | ((bp_acts & 2) >> 1);
		act_selector conv_clear_acts_diff = clear_acts_diff & 1;
		weight_selector conv_clear_weight_diff = ((clear_weights_diff & 1) << 1) | ((clear_acts_diff & 2) >> 1);
		conv_->backward(conv_act_selector, conv_weight_selector, conv_clear_acts_diff, conv_clear_weight_diff);
	}
}