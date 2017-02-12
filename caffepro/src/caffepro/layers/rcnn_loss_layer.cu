
#include <caffepro/layers/rcnn_loss_layer.h>
#include <caffepro/layers/softmax_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

#define MAX_MINIBATCH 512

namespace caffepro {
	rcnn_loss_layer::rcnn_loss_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 2;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_SAME_NUM
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);

		attr_.usage = layer_attribute::USAGE_LOSS;
		attr_.device_dispatcher_forward = layer_attribute::INPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::INPUT_BASE;
	}

	rcnn_loss_layer::~rcnn_loss_layer() {
		release_all();
	}

	void rcnn_loss_layer::init() {
		check_input();

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
		avg_prob_dev_.reset(new node_blob());
	}

	void rcnn_loss_layer::resize() {
		check_input();

		softmax_->resize();
		CHECK(inputs_[0]->same_dim_at(2));

		int n_devices = (int)inputs_[0]->size();

		if (outputs_[0]->size() == 0) {
			outputs_[0]->tags().resize(7);
			outputs_[0]->tags()[0] = "Error";
			outputs_[0]->tags()[1] = "Loss";
			outputs_[0]->tags()[2] = "Foreground Images per Batch";
			outputs_[0]->tags()[3] = "Foreground Correct Images per Batch";
			outputs_[0]->tags()[4] = "Background Images per Batch";
			outputs_[0]->tags()[5] = "Background Correct Images per Batch";
			outputs_[0]->tags()[6] = "False negtive";

			for (int nd = 0; nd < n_devices; nd++) {
				CHECK(inputs_[1]->get(nd)->count() == inputs_[0]->get(nd)->num());

				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, 1, (int)outputs_[0]->tags().size(), 1, 1, inputs_[0]->get(nd)->device_id())
					));

				avg_prob_dev_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
						context_,
						inputs_[0]->get(nd)->num(),
						inputs_[0]->get(nd)->dim_at(2),
						1,
						1,
						inputs_[0]->get(nd)->device_id()
						)
					));

				correct_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
						context_, inputs_[0]->get(nd)->num(), 1, 1, 1, inputs_[0]->get(nd)->device_id()
						)
					));

				sum_multiplier_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
						context_,
						1,
						1,
						inputs_[0]->get(nd)->height(),
						inputs_[0]->get(nd)->width(),
						inputs_[0]->get(nd)->device_id()
						)
					));

				sum_multiplier_->get(nd)->fill_data((data_type)1.f);
			}
		}
		else if (inputs_[0]->reshaped()) {
			// do not need to reshape top

			for (int nd = 0; nd < n_devices; nd++) {
				CHECK(inputs_[1]->get(nd)->count() == inputs_[0]->get(nd)->num());

				avg_prob_dev_->get(nd)->reshape_4d(
					inputs_[0]->get(nd)->num(),
					inputs_[0]->get(nd)->channels(),
					1,
					1
					);

				correct_->get(nd)->reshape_4d(inputs_[0]->get(nd)->num(), 1, 1, 1);

				sum_multiplier_->get(nd)->reshape_4d(1, 1, inputs_[0]->get(nd)->height(), inputs_[0]->get(nd)->width());
				sum_multiplier_->get(nd)->fill_data((data_type)1.f);
			}
		}
	}

	__global__ static void calc_acc_gpu_kernel(const int n, const int dim, const data_type *probs,
		const data_type *label, data_type *correct, data_type *stat) {
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
			int frg = 0, bkg = 0;
			int frg_acc = 0, bkg_acc = 0;
			int false_neg = 0;
			for (int i = 0; i < n; i++) {
				n_correct += ans[i];
				ls += ans_loss[i];
				
				int cur_ground_truth = (label[i] >= 0 ? (int)(label[i] + 0.5) : -1);
				if (cur_ground_truth >= 0) {
					if (cur_ground_truth == dim - 1) { // background
						bkg++;
						if (ans[i]) bkg_acc++;
					}
					else { // foreground
						frg++;
						if (ans[i]) frg_acc++;
						else if ((int)correct[i] == dim - 1) false_neg++;
					}
				}
			}
			stat[0] = n_correct;			// number of correct images
			stat[1] = ls;					// sum loss
			stat[2] = (data_type)frg;		// foreground images
			stat[3] = (data_type)frg_acc;	// foreground correct images
			stat[4] = (data_type)bkg;		// background images
			stat[5] = (data_type)bkg_acc;	// background correct images
			stat[6] = (data_type)false_neg; // false negtive
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

	void rcnn_loss_layer::on_before_forward() {
		// The forward pass computes the softmax prob values.
		softmax_->forward();
	}

	void rcnn_loss_layer::on_forward(int device_index) {
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
				avg_prob_dev_->get(device_index)->mutable_gpu_data()
				);
		}
		else {
			avg_prob_dev_->get(device_index)->copy_data_from_via_gpu(*prob_->get(device_index));
		}

		CHECK_LE(num, MAX_MINIBATCH);

		calc_acc_gpu_kernel<<<1, num>>>(
			num,
			feature_dim,
			avg_prob_dev_->get(device_index)->gpu_data(),
			inputs_[1]->get(device_index)->gpu_data(),
			correct_->get(device_index)->mutable_gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data()
			);
	}

	void rcnn_loss_layer::on_after_forward() {
		// merge result to the first device
		int n_devices = (int)outputs_[0]->size();
		data_type *result_data = outputs_[0]->get(0)->mutable_cpu_data();
		int count = outputs_[0]->get(0)->count();

		for (int nd = 1; nd < n_devices; nd++) {
			const data_type *dev_result_data = outputs_[0]->get(nd)->cpu_data();
			for (int i = 0; i < count; i++) {
				result_data[i] += dev_result_data[i];
			}
		}

		int num = inputs_[0]->sum_num();
		result_data[0] = 1 - result_data[0] / num;	// error
		result_data[1] /= num;						// loss
	}

	void rcnn_loss_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
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