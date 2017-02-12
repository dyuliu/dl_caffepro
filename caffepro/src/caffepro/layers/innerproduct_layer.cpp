
#include <caffepro/layers/innerproduct_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <boost/scoped_ptr.hpp>

namespace caffepro {
	innerproduct_layer::innerproduct_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_INNER_COUNT_ACROSS_DEVICES
			);
	}

	innerproduct_layer::~innerproduct_layer() {
		release_all();
	}

	void innerproduct_layer::init() {
		check_input();

		num_outputs_ = layer_param_.inner_product_param().num_output();
		bias_term_ = layer_param_.inner_product_param().bias_term();
		num_inputs_ = inputs_[0]->get(0)->inner_count();
		
		// set weights
		if (bias_term_) {
			weights_.resize(2);
		}
		else {
			weights_.resize(1);
		}

		int n_devices = (int)inputs_[0]->size();
		boost::scoped_ptr<device_blob> weight_template(
			device_blob::create_4d(context_, num_outputs_, num_inputs_, 1, 1));
		weights_[0].reset(new node_blob());
		weights_[0]->add_like(context_, *weight_template, *inputs_[0]); 
		boost::scoped_ptr<filler> weight_filler(filler::create(context_, layer_param_.inner_product_param().weight_filler()));
		weight_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		if (bias_term_) {
			boost::scoped_ptr<device_blob> bias_template(
				device_blob::create_4d(context_, 1, num_outputs_, 1, 1)
				);
			weights_[1].reset(new node_blob());
			weights_[1]->add_like(context_, *bias_template, *inputs_[0]); // bias should be unmanaged
			boost::scoped_ptr<filler> bias_filler(filler::create(context_, layer_param_.inner_product_param().bias_filler()));
			bias_filler->fill(*weights_[1]->get(0));
			weights_[1]->broadcast_data_via_gpu(0);

			bias_multiplier_.reset(new node_blob());
		}
	}

	void innerproduct_layer::resize() {
		check_input();
		CHECK_EQ(inputs_[0]->get(0)->inner_count(), num_inputs_);
		int n_devices = (int)inputs_[0]->size();

		if (outputs_[0]->size() == 0) {
			for (int nd = 0; nd < n_devices; nd++) {
				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
						inputs_[0]->get(nd)->context(),
						inputs_[0]->get(nd)->num(),
						num_outputs_,
						1,
						1,
						inputs_[0]->get(nd)->device_id()
						)
					));

				if (bias_term_) {
					bias_multiplier_->add(boost::shared_ptr<device_blob>(
						device_blob::create_4d(
							inputs_[0]->get(nd)->context(),
							inputs_[0]->get(nd)->num(),
							1, 1, 1,
							inputs_[0]->get(nd)->device_id()
							)
						));
					bias_multiplier_->get(nd)->fill_data((data_type)1);
				}
			}
		}
		else {
			CHECK_EQ(inputs_[0]->size(), outputs_[0]->size());

			for (int nd = 0; nd < n_devices; nd++) {
				if (inputs_[0]->get(nd)->reshaped()
					&& inputs_[0]->get(nd)->num() != outputs_[0]->get(nd)->num()) {
					outputs_[0]->get(nd)->reshape_4d(
						inputs_[0]->get(nd)->num(),
						num_outputs_,
						1,
						1
						);

					if (bias_term_) {
						bias_multiplier_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), 1, 1, 1);
						bias_multiplier_->get(nd)->fill_data((data_type)1);
					}
				}
			}
		}
	}

	void innerproduct_layer::on_forward(int device_index) {
		const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();
		data_type* top_data = outputs_[0]->get(device_index)->mutable_gpu_data();
		const data_type* weight = weights_[0]->get(device_index)->gpu_data();
		cublas_wrapper<data_type> cublas(context_, inputs_[0]->get(device_index)->device_id());
		int num = inputs_[0]->get(device_index)->num();

		cublas.gemm(CblasNoTrans, CblasTrans, 
			num, num_outputs_, num_inputs_, 
			(data_type)1.f, bottom_data, weight, 
			(data_type)0.f, top_data);

		if (bias_term_) {
			cublas.gemm(CblasNoTrans, CblasNoTrans, 
				num, num_outputs_, 1, 
				(data_type)1.f, bias_multiplier_->get(device_index)->gpu_data(), weights_[1]->get(device_index)->gpu_data(), 
				(data_type)1.f, top_data);
		}
	}

	void innerproduct_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type alpha = 1;
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		const data_type beta_weight = get_beta(clear_weights_diff, 0);
		const data_type beta_bias = get_beta(clear_weights_diff, 1);

		cublas_wrapper<data_type> cublas(context_, inputs_[0]->get(device_index)->device_id());
		int num = inputs_[0]->get(device_index)->num();

		if (should_bp(bp_weights, 0)) {
			// Gradient with respect to weight
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			const data_type* bottom_data = inputs_[0]->get(device_index)->gpu_data();

			cublas.gemm(CblasTrans, CblasNoTrans, 
				num_outputs_, num_inputs_, num, 
				alpha, top_diff, bottom_data, 
				beta_weight, weights_[0]->get(device_index)->mutable_gpu_diff());
		}

		if (bias_term_ && should_bp(bp_weights, 1)) {
			// Gradient with respect to bias
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			cublas.gemv(CblasTrans, num, num_outputs_, 
				alpha, top_diff, bias_multiplier_->get(device_index)->gpu_data(), 
				beta_bias, weights_[1]->get(device_index)->mutable_gpu_diff());
		}

		if (should_bp(bp_acts, 0)) {
			// Gradient with respect to bottom data
			const data_type* top_diff = outputs_[0]->get(device_index)->gpu_diff();
			cublas.gemm(CblasNoTrans, CblasNoTrans, num, num_inputs_, num_outputs_, 
				alpha, top_diff, weights_[0]->get(device_index)->gpu_data(), 
				beta_acts, inputs_[0]->get(device_index)->mutable_gpu_diff());
		}
	}
}