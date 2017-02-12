
#include <caffepro/layers/scalebias_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filler.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

#include <caffepro/context/common_names.h>
#include <caffepro/solver/sgd_solver.h>

#include <boost/scoped_ptr.hpp>

namespace caffepro {
	scalebias_layer::scalebias_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {
		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			//| layer_attribute::CF_REQUIRE_NDIM_4
			//| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);
	}

	scalebias_layer::~scalebias_layer() {
		release_all();
	}

	void scalebias_layer::init() {
		check_input();

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		const int channels = inputs_[0]->dim_at(2);

		// set weights
		weights_.resize(2);
		
		int n_devices = (int)inputs_[0]->size();
		boost::scoped_ptr<device_blob> weight_template(device_blob::create_4d(context_, 1, channels, 1, 1));
		weights_[0].reset(new node_blob());
		weights_[0]->add_like(context_, *weight_template, *inputs_[0]);
		boost::scoped_ptr<filler> weight_filler(filler::create(context_, layer_param_.scalebias_param().weight_filler()));
		weight_filler->fill(*weights_[0]->get(0));
		weights_[0]->broadcast_data_via_gpu(0);

		boost::scoped_ptr<device_blob> bias_template(device_blob::create_4d(context_, 1, channels, 1, 1));
		weights_[1].reset(new node_blob());
		weights_[1]->add_like(context_, *bias_template, *inputs_[0]); // bias should be unmanaged
		boost::scoped_ptr<filler> bias_filler(filler::create(context_, layer_param_.scalebias_param().bias_filler()));
		bias_filler->fill(*weights_[1]->get(0));
		weights_[1]->broadcast_data_via_gpu(0);

		sum_multiplier_.reset(new node_blob());
		sum_multiplier_num_.reset(new node_blob());

	}

	void scalebias_layer::resize() {
		caffepro_layer::resize();

		CHECK(inputs_[0]->same_dim_at(2)); // same channel
		const int channels = inputs_[0]->dim_at(2);

		int n_devices = (int)inputs_[0]->size();

		if (sum_multiplier_->size() == 0) { // first run
			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				int spatial_size = inputs_[0]->get(nd)->width() * inputs_[0]->get(nd)->height();

				sum_multiplier_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, 1, spatial_size, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				sum_multiplier_num_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, inputs_[0]->get(nd)->num(), 1, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				sum_multiplier_->get(nd)->fill_data(1.f);
				sum_multiplier_num_->get(nd)->fill_data(1.f);
			}
		}
		else {
			for (int nd = 0; nd < (int)inputs_[0]->size(); nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					int num = inputs_[0]->get(nd)->num();

					int spatial_size = inputs_[0]->get(nd)->width() * inputs_[0]->get(nd)->height();
					if (spatial_size != sum_multiplier_->get(nd)->count()) {
						sum_multiplier_->get(nd)->reshape_4d(1, spatial_size, 1, 1);
						sum_multiplier_->get(nd)->fill_data(1.f);
					}
				}
			}
		}
	}

	__global__ static void scalebias_forward(const int n, const int spatial_size, const int channels,
		const data_type *bottom_data,
		const data_type *scale_data, const data_type *shift_data,
		data_type *top_data) {

		CUDA_KERNEL_LOOP(index, n) {
			const int c = (index / spatial_size) % channels;

			top_data[index] = bottom_data[index] * scale_data[c] + shift_data[c];
		}
	}

	void scalebias_layer::on_forward(int device_index) {
		int num = inputs_[0]->get(device_index)->num();
		int channels = inputs_[0]->get(device_index)->channels();
		int spatial_size = inputs_[0]->get(device_index)->height() * inputs_[0]->get(device_index)->width();
	
		// do forward
		int count = inputs_[0]->get(device_index)->count();
		KERNEL_CALL(scalebias_forward, count)(
			count, spatial_size, channels,
			inputs_[0]->get(device_index)->gpu_data(),
			weights_[0]->get(device_index)->gpu_data(), weights_[1]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data()
			);

		CUDA_POST_KERNEL_CHECK;
	}

	__global__ static void scalebias_backward(const int n, const int spatial_size, const int channels,
		const data_type *top_diff,
		const data_type *scale_data,
		data_type *bottom_diff) {

		CUDA_KERNEL_LOOP(index, n) {
			const int c = (index / spatial_size) % channels;

			bottom_diff[index] = top_diff[index] * scale_data[c];
		}
	}

	void scalebias_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		data_type beta_weight = get_beta(clear_weights_diff, 0);
		data_type beta_bias = get_beta(clear_weights_diff, 1);

		if (!should_bp(bp_weights, 0)) {
			beta_weight = 0.f;
		}

		if (!should_bp(bp_weights, 1)) {
			beta_bias = 0.f;
		}

		cublas_wrapper<data_type> cublas(context_, inputs_[0]->get(device_index)->device_id());
		
		int num = inputs_[0]->get(device_index)->num();
		int channels = inputs_[0]->get(device_index)->channels();
		int num_of_vecs = num * channels;
		int spatial_size = inputs_[0]->get(device_index)->height() * inputs_[0]->get(device_index)->width();

		data_type *tmp_diff = reinterpret_cast<data_type*>(context_->get_current_device()->memory()->allocate(num_of_vecs * sizeof(data_type)));

		// Gradient with respect to bias
		cublas.gemv(
			CblasNoTrans,
			num_of_vecs,
			spatial_size,
			(data_type)1.f,
			outputs_[0]->get(device_index)->gpu_diff(),
			sum_multiplier_->get(device_index)->gpu_data(),
			(data_type)0.f,
			tmp_diff
			); // cache sum_spatial(top_diff) into EX_diff

		cublas.gemv(
			CblasTrans,
			num,
			channels,
			(data_type)1.f,
			tmp_diff,
			sum_multiplier_num_->get(device_index)->gpu_data(),
			beta_bias,
			weights_[1]->get(device_index)->mutable_gpu_diff()
			); // bp bias, i.e. bias_diff = sum_num(sum_spatial(top_diff))

		// Gradient with respect to weight
		cublas.mul(outputs_[0]->get(device_index)->count(), outputs_[0]->get(device_index)->gpu_diff(), inputs_[0]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data()); // cache (top_diff .* input) into top_data

		cublas.gemv(
			CblasNoTrans,
			num_of_vecs,
			spatial_size,
			(data_type)1.f,
			outputs_[0]->get(device_index)->gpu_data(),
			sum_multiplier_->get(device_index)->gpu_data(),
			(data_type)0.f,
			tmp_diff
			); // cache sum_spatial(top_diff .* input) into EX_diff

		cublas.gemv(
			CblasTrans,
			num,
			channels,
			(data_type)1.f,
			tmp_diff,
			sum_multiplier_num_->get(device_index)->gpu_data(),
			beta_weight,
			weights_[0]->get(device_index)->mutable_gpu_diff()
			); // bp scale, i.e. scale_diff = sum_num(sum_spatial(top_diff .* input))

		// Gradient with respect to bottom data
		int count = inputs_[0]->get(device_index)->count();
		KERNEL_CALL(scalebias_backward, count)(
			count, spatial_size, channels,
			outputs_[0]->get(device_index)->gpu_diff(),
			weights_[0]->get(device_index)->gpu_data(),
			inputs_[0]->get(device_index)->mutable_gpu_diff()
			); // bottom_diff = top_diff * scale
		CUDA_POST_KERNEL_CHECK;

		context_->get_current_device()->memory()->free(tmp_diff);
	}
}