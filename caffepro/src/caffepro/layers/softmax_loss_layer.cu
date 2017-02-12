
#include <caffepro/layers/softmax_loss_layer.h>
#include <caffepro/layers/softmax_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/string_uitls.h>

#include <functional>

#define MAX_MINIBATCH 512

namespace caffepro {
	softmax_loss_layer::softmax_loss_layer(caffepro_context *context, const LayerParameter &param)
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

	softmax_loss_layer::~softmax_loss_layer() {
		release_all();
	}

	void softmax_loss_layer::init() {
		check_input();
		output_top_n_ = 0;

		coeff_ = (data_type)1.f;
		if (layer_param_.has_loss_param() && layer_param_.loss_param().has_coeff()) {
			coeff_ = layer_param_.loss_param().coeff();
		}

		softmax_inputs_.resize(1);
		softmax_inputs_[0] = inputs_[0];
		prob_.reset(new node_blob());
		softmax_outputs_.resize(1);
		softmax_outputs_[0] = prob_;
		
		softmax_.reset(new softmax_layer(context_, layer_param_));
		softmax_->bind(softmax_inputs_, softmax_outputs_);
		softmax_->init();

		correct_.reset(new node_blob());
		sum_multiplier_.reset(new node_blob());
		avg_prob_.reset(new node_blob());
	}

	void softmax_loss_layer::resize() {
		check_input();

		softmax_->resize();
		CHECK(inputs_[0]->get(0)->same_dim_at(2));
		CHECK(inputs_[1]->get(0)->count() == inputs_[0]->get(0)->num());

		if (outputs_[0]->size() == 0) {

			int num_of_outputs = 2;
			if (!context_->get_global_cfg(GLOBALCFGNAME_SHOW_TOP_N_ACC).empty()) {
				num_of_outputs++;
				output_top_n_ = string_to_int(context_->get_global_cfg(GLOBALCFGNAME_SHOW_TOP_N_ACC));
				CHECK_GT(output_top_n_, 0);
			}

			outputs_[0]->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, 1, num_of_outputs, 1, 1, inputs_[0]->get(0)->device_id())
				));

			outputs_[0]->tags().resize(num_of_outputs);
			outputs_[0]->tags()[0] = "Error";
			outputs_[0]->tags()[1] = "Loss";
			if (num_of_outputs >= 3) {
				outputs_[0]->tags()[2] = "Top " + std::to_string(output_top_n_) + " error";
			}

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

	__global__ static void calc_acc_gpu_kernel(const int n, const int dim, const data_type *probs,
		const data_type *label, data_type *correct, data_type *acc, data_type *loss) {
		__shared__ int ans[MAX_MINIBATCH];
		__shared__ data_type ans_loss[MAX_MINIBATCH];

		int cur_ins = threadIdx.x;
		probs += cur_ins * dim;

		data_type max_v = -1;
		int max_index = -1;
		for (int i = 0; i < dim; i++) {
			if (probs[i] > max_v) {
				max_v = probs[i];
				max_index = i;
			}
		}

		int ground_truth = (label[cur_ins] >= 0 ? (int)(label[cur_ins] + 0.5) : -1);
		correct[cur_ins] = max_index;
		ans[cur_ins] = (ground_truth == max_index) ? 1 : 0;
		ans_loss[cur_ins] = (ground_truth >= 0 ? -__logf(probs[ground_truth]) : 0);

		__syncthreads();

		if (cur_ins == 0) {
			data_type n_correct = 0;
			data_type ls = 0;
			for (int i = 0; i < n; i++) {
				n_correct += ans[i];
				ls += ans_loss[i];
			}
			acc[0] = 1 - n_correct / n;
			loss[0] = ls / n;
		}
	}

	__global__ void static bp_softlost_gpu_kernel(const int n, const int feature_dim, const int spatial_size,
		const data_type *probs, const data_type *label,
		data_type *diff, const data_type coeff, const data_type scale_targets) {

		CUDA_KERNEL_LOOP(index, n) {
			int dest_idx = index;

			index /= spatial_size;
			int cur_ins = index / feature_dim;
			int cur_dim = index % feature_dim;
			int total_ins = n / feature_dim / spatial_size;

			int ground_truth = (label[cur_ins] >= 0 ? (int)(label[cur_ins] + 0.5) : -1);

			if (ground_truth >= 0) {
				if (ground_truth == cur_dim) {
					data_type v = coeff * (probs[dest_idx] - 1) / (data_type)total_ins / (data_type)spatial_size;
					if (scale_targets == 0) {
						diff[dest_idx] = v;
					}
					else {
						diff[dest_idx] = diff[dest_idx] * scale_targets + v;
					}
				}
				else {
					data_type v = coeff * probs[dest_idx] / (data_type)total_ins / (data_type)spatial_size;
					if (scale_targets == 0) {
						diff[dest_idx] = v;
					}
					else {
						diff[dest_idx] = diff[dest_idx] * scale_targets + v;
					}
				}
			}
			else {
				if (scale_targets == 0) {
					diff[dest_idx] = 0;
				}
				// do not need to modify the diff when scale_target != 0
			}
		}
	}

	void softmax_loss_layer::on_forward(int device_index) {
		// The forward pass computes the softmax prob values.
		softmax_->forward();

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

		CHECK_LE(num, MAX_MINIBATCH);

		//auto ptr = avg_prob_->get(device_index)->cpu_data();
		//auto ptr2 = inputs_[1]->get(device_index)->cpu_data();

		calc_acc_gpu_kernel<<<1, num>>>(
			num, 
			feature_dim, 
			avg_prob_->get(device_index)->gpu_data(),
			inputs_[1]->get(device_index)->gpu_data(), 
			correct_->get(device_index)->mutable_gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data(), 
			outputs_[0]->get(device_index)->mutable_gpu_data() + 1
			);

		// optional: calc top_n error
		// cpu code
		if (output_top_n_ >= 1) {
			const data_type *avg_prob_data = avg_prob_->get(device_index)->cpu_data();
			const data_type *label_data = inputs_[1]->get(device_index)->cpu_data();
			std::vector<std::pair<data_type, int> > score(feature_dim);

			int top_n_acc = 0;
			for (int i = 0; i < num; i++, avg_prob_data += feature_dim) {
				for (int k = 0; k < feature_dim; k++) {
					score[k] = std::make_pair(avg_prob_data[k], k);
				}
				std::sort(score.begin(), score.end(), std::greater<std::pair<data_type, int> >());

				int label_id = (int)(label_data[i] + 0.5f);
				for (int k = 0; k < output_top_n_ && k < feature_dim; k++) {
					if (score[k].second == label_id) {
						top_n_acc++;
						break;
					}
				}
			}

			outputs_[0]->get(device_index)->mutable_cpu_data()[2] = 1.f - (data_type)top_n_acc / num;
		}
	}

	void softmax_loss_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);
		cudnnHandle_t handle = context_->get_current_device()->cudnn_handle();

		if (should_bp(bp_acts, 0)) {
			const int count = prob_->get(device_index)->count();
			const int feature_dim = prob_->get(device_index)->channels();
			const int spatial_size = prob_->get(device_index)->width() * prob_->get(device_index)->height();

			KERNEL_CALL(bp_softlost_gpu_kernel, count)(
				count,
				feature_dim,
				spatial_size,
				prob_->get(device_index)->gpu_data(),
				inputs_[1]->get(device_index)->gpu_data(),
				inputs_[0]->get(device_index)->mutable_gpu_diff(),
				coeff_,
				beta_acts
				);
		}
	}
}