
#include <caffepro/layers/matrix_mul_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	matrix_mul_layer::matrix_mul_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_DIMTYPE
			| layer_attribute::CF_REQUIRE_SAME_COUNT_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_SAME_INNER_COUNT_ACROSS_DEVICES
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM 		// remove it in the future
			);
	}

	matrix_mul_layer::~matrix_mul_layer() {
		release_all();
	}

	void matrix_mul_layer::init() {
		check_input();

		trans_A_ = layer_param_.matrix_mul_param().trans_a();
		trans_B_ = layer_param_.matrix_mul_param().trans_b();
	}

	void matrix_mul_layer::resize() {
		check_input();

		int r1 = inputs_[0]->get(0)->num();
		int c1 = inputs_[0]->get(0)->inner_count();
		if (trans_A_) std::swap(r1, c1);
		int r2 = inputs_[1]->get(0)->num();
		int c2 = inputs_[1]->get(0)->inner_count();
		if (trans_B_) std::swap(r2, c2);
		CHECK_EQ(c1, r2);

		M_ = r1, N_ = c2, K_ = c1;

		int n_devices = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_devices; nd++) {
			if (inputs_[0]->get(nd)->reshaped() || inputs_[1]->get(nd)->reshaped()) {
				if (!trans_B_) {
					outputs_[0]->set_4d(
						nd,
						r1,
						inputs_[1]->get(nd)->channels(),
						inputs_[1]->get(nd)->height(),
						inputs_[1]->get(nd)->width(),
						inputs_[1]->get(nd)->device_id(),
						context_
						);
				}
				else {
					outputs_[0]->set_4d(nd, r1, c2, 1, 1, inputs_[1]->get(nd)->device_id(), context_);
				}
			}
		}
	}

	inline CBLAS_TRANSPOSE get_trans_mark(bool trans) {
		return trans ? CblasTrans : CblasNoTrans;
	}

	void matrix_mul_layer::on_forward(int device_index) {
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		auto transA = get_trans_mark(trans_A_);
		auto transB = get_trans_mark(trans_B_);

		cublas.gemm(transA, transB, M_, N_, K_, (data_type)1.f,
			inputs_[0]->get(device_index)->gpu_data(),
			inputs_[1]->get(device_index)->gpu_data(),
			(data_type)0.f,
			outputs_[0]->get(device_index)->mutable_gpu_data());
	}

	void matrix_mul_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		const data_type *data_A = inputs_[0]->get(device_index)->gpu_data();
		const data_type *data_B = inputs_[1]->get(device_index)->gpu_data();
		const data_type *diff = outputs_[0]->get(device_index)->gpu_diff();
		data_type *diff_A = inputs_[0]->get(device_index)->mutable_gpu_diff();
		data_type *diff_B = inputs_[1]->get(device_index)->mutable_gpu_diff();

		if (should_bp(bp_acts, 0)) {
			data_type beta = get_beta(clear_acts_diff, 0);

			if (!trans_A_) {
				cublas.gemm(CblasNoTrans, get_trans_mark(!trans_B_), M_, K_, N_, 1.f,
					diff, data_B, beta, diff_A);
			}
			else {
				cublas.gemm(get_trans_mark(trans_B_), CblasTrans, K_, M_, N_, 1.f,
					data_B, diff, beta, diff_A);
			}
		}

		if (should_bp(bp_acts, 1)) {
			data_type beta = get_beta(clear_acts_diff, 1);
		
			if (!trans_B_) {
				cublas.gemm(get_trans_mark(!trans_A_), CblasNoTrans, K_, N_, M_, 1.f,
					data_A, diff, beta, diff_B);
			}
			else {
				cublas.gemm(CblasTrans, get_trans_mark(trans_A_), N_, K_, M_, 1.f,
					diff, data_A, beta, diff_B);
			}
		}
	}
}