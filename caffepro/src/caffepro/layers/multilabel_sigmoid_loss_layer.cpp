
#include <caffepro/layers/multilabel_sigmoid_loss_layer.h>
#include <caffepro/layers/sigmoid_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/utils/accuracy.h>

namespace caffepro {
	multilabel_sigmoid_loss_layer::multilabel_sigmoid_loss_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_UNIQUE_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_NUM
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);

		attr_.usage = layer_attribute::USAGE_LOSS;
	}

	multilabel_sigmoid_loss_layer::~multilabel_sigmoid_loss_layer() {
		release_all();
	}

	void multilabel_sigmoid_loss_layer::init() {
		check_input();

		coeff_ = (data_type)1.f;
		if (layer_param_.has_loss_param() && layer_param_.loss_param().has_coeff()) {
			coeff_ = layer_param_.loss_param().coeff();
		}

		sigmoid_inputs_.resize(1);
		sigmoid_inputs_[0] = inputs_[0];
		prob_.reset(new node_blob());
		sigmoid_outputs_.resize(1);
		sigmoid_outputs_[0] = prob_;

		sigmoid_.reset(new sigmoid_layer(context_, layer_param_));
		sigmoid_->bind(sigmoid_inputs_, sigmoid_outputs_);
		sigmoid_->init();

		correct_.reset(new node_blob());
		sum_multiplier_.reset(new node_blob());
		avg_prob_.reset(new node_blob());
	}

	void multilabel_sigmoid_loss_layer::resize() {
		check_input();

		sigmoid_->resize();
		CHECK(inputs_[0]->get(0)->same_dim_at(2));
		CHECK(inputs_[1]->get(0)->same_dim_at(2));
		CHECK(inputs_[1]->get(0)->dim_at(2) == inputs_[0]->get(0)->dim_at(2));

		if (outputs_[0]->size() == 0) {
			outputs_[0]->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, 1, 2, 1, 1, inputs_[0]->get(0)->device_id())
				));

			outputs_[0]->tags().resize(2);
			outputs_[0]->tags()[0] = "Error";
			outputs_[0]->tags()[1] = "Loss";

			avg_prob_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
				context_,
				inputs_[0]->get(0)->num(),
				inputs_[0]->get(0)->dim_at(2),
				1,
				1,
				inputs_[0]->get(0)->device_id()
				)
				));

			correct_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
				context_, inputs_[0]->get(0)->num(), 1, 1, 1, inputs_[0]->get(0)->device_id()
				)
				));

			sum_multiplier_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
				context_,
				1,
				1,
				inputs_[0]->get(0)->height(),
				inputs_[0]->get(0)->width(),
				inputs_[0]->get(0)->device_id()
				)
				));

			sum_multiplier_->get(0)->fill_data((data_type)1.f);
		}
		else if (inputs_[0]->reshaped()) {
			// do not need to reshape top

			avg_prob_->get(0)->reshape_4d(
				inputs_[0]->get(0)->num(),
				inputs_[0]->get(0)->channels(),
				1,
				1
				);

			correct_->get(0)->reshape_4d(inputs_[0]->get(0)->num(), 1, 1, 1);

			sum_multiplier_->get(0)->reshape_4d(1, 1, inputs_[0]->get(0)->height(), inputs_[0]->get(0)->width());
			sum_multiplier_->get(0)->fill_data((data_type)1.f);
		}
	}

	void multilabel_sigmoid_loss_layer::on_forward(int device_index) {
		// The forward pass computes the softmax prob values.
		sigmoid_->forward();

		// average up
		const int spatial_size = inputs_[0]->get(device_index)->width() * inputs_[0]->get(device_index)->height();
		const int num = inputs_[0]->get(device_index)->num();
		const int feature_dim = inputs_[0]->get(device_index)->dim_at(2);

		CHECK_GT(spatial_size, 0);

		if (spatial_size > 1) {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
			cublas.gemv(
				CblasNoTrans,
				num * feature_dim,
				spatial_size,
				(data_type)1. / (data_type)spatial_size,
				prob_->get(device_index)->gpu_data(),
				sum_multiplier_->get(device_index)->gpu_data(),
				(data_type)0.,
				avg_prob_->get(device_index)->mutable_gpu_data()
				);
		}
		else {
			avg_prob_->get(device_index)->copy_data_from_via_gpu(*prob_->get(device_index));
		}

		float accuracy = 0, loss = 0;
		compute_multilabel_accurancy(avg_prob_->get(device_index).get(), inputs_[1]->get(device_index).get(), &accuracy, &loss);
		outputs_[0]->get(device_index)->mutable_cpu_data()[0] = 1 - accuracy;
		outputs_[0]->get(device_index)->mutable_cpu_data()[1] = loss;
	}

	void multilabel_sigmoid_loss_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();

		if (should_bp(bp_acts, 0)) {
			const int spatial_size = inputs_[0]->get(device_index)->width() * inputs_[0]->get(device_index)->height();
			const int count = inputs_[0]->get(device_index)->count();
			const int num = inputs_[0]->get(device_index)->num();
			const int dim = inputs_[0]->get(device_index)->channels();

			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

			// First, compute the diff 
			data_type* sigmoid_output_data = prob_->get(device_index)->mutable_gpu_data();
			const data_type* target = inputs_[1]->get(device_index)->gpu_data();
			data_type* bottom_diff = inputs_[0]->get(device_index)->mutable_gpu_diff();

			cublas.gemm(CblasNoTrans, CblasNoTrans, num * dim, spatial_size, 1, (data_type)-1.f,
				target, sum_multiplier_->get(device_index)->gpu_data(), (data_type)1.f, sigmoid_output_data);

			if (beta_acts == 0) {
				cublas.scale(count, coeff_ / (data_type)num / (data_type)spatial_size, sigmoid_output_data, bottom_diff);
			}
			else {
				cublas.axpby(count, coeff_ / (data_type)num / (data_type)spatial_size, sigmoid_output_data, beta_acts, bottom_diff);
			}
		}
	}
}