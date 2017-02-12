
#include <caffepro/layers/softmax_ohem_layer.h>
#include <caffepro/layers/softmax_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/string_uitls.h>

#include <functional>

#define MAX_MINIBATCH 512

namespace caffepro {
	softmax_ohem_layer::softmax_ohem_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 3; // 0: fc1000; 1: labels; 2: data

		attr_.num_outputs_min = attr_.num_outputs_max = 3; // 0: loss; 1: selected labels; 2: selected data;

		attr_.set_constraint(
			//layer_attribute::CF_REQUIRE_UNIQUE_DEVICE
			//layer_attribute::CF_REQUIRE_SAME_NUM
			//| layer_attribute::CF_REQUIRE_SAME_DEVICE
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);

		attr_.usage = layer_attribute::USAGE_LOSS;
	}

	softmax_ohem_layer::~softmax_ohem_layer() {
		release_all();
	}

	void softmax_ohem_layer::init() {
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

		bp_indicator_.reset(new node_blob());

		ohem_size_ = layer_param_.softmaxohem_param().ohem_size();
		force_random_ = layer_param_.softmaxohem_param().force_random();

		use_max_loss_ = layer_param_.softmaxohem_param().use_max_loss();
	}

	void softmax_ohem_layer::resize() {
		check_input();

		softmax_->resize();

		CHECK_EQ(inputs_[0]->size(), 1);
		CHECK(inputs_[0]->get(0)->same_dim_at(2));
		CHECK(inputs_[1]->get(0)->count() == inputs_[0]->get(0)->num());

		CHECK_LE(ohem_size_, inputs_[0]->get(0)->num());

		// verify
		int num = 0;
		for (int i = 0; i < inputs_[2]->size(); i++)
			num += inputs_[2]->get(i)->num();

		CHECK_EQ(num, inputs_[0]->get(0)->num());
		CHECK_EQ(ohem_size_ % inputs_[2]->size(), 0);

		if (outputs_[0]->size() == 0) {

			int num_of_outputs = 4;

			if (!context_->get_global_cfg(GLOBALCFGNAME_SHOW_TOP_N_ACC).empty()) {
				num_of_outputs += 2;
				output_top_n_ = string_to_int(context_->get_global_cfg(GLOBALCFGNAME_SHOW_TOP_N_ACC));
				CHECK_GT(output_top_n_, 0);
			}

			outputs_[0]->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, 1, num_of_outputs, 1, 1, inputs_[0]->get(0)->device_id())
				));

			outputs_[0]->tags().resize(num_of_outputs);
			outputs_[0]->tags()[0] = "Error";
			outputs_[0]->tags()[1] = "Loss";
			outputs_[0]->tags()[2] = "OHEM Error";
			outputs_[0]->tags()[3] = "OHEM Loss";
			if (num_of_outputs >= 5) {
				outputs_[0]->tags()[4] = "Top " + std::to_string(output_top_n_) + " error";
				outputs_[0]->tags()[5] = "OHEM Top " + std::to_string(output_top_n_) + " error";
			}

			outputs_[1]->set_4d(0,
				ohem_size_, inputs_[1]->get(0)->channels(), inputs_[1]->get(0)->height(), inputs_[1]->get(0)->width(),
				inputs_[1]->get(0)->device_id(),
				context_);

			for (int nd = 0; nd < inputs_[2]->size(); nd++) {
				outputs_[2]->set_4d(nd,
					ohem_size_ / inputs_[2]->size(), inputs_[2]->get(nd)->channels(), inputs_[2]->get(nd)->height(), inputs_[2]->get(nd)->width(),
					inputs_[2]->get(nd)->device_id(),
					context_);
			}

			bp_indicator_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
				context_, inputs_[0]->get(0)->num(), 1, 1, 1, inputs_[0]->get(0)->device_id()
				)
				));

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

			bp_indicator_->get(0)->reshape_4d(
				inputs_[0]->get(0)->num(), 1, 1, 1
				);

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

	void softmax_ohem_layer::on_forward(int device_index) {
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

		const int tag_offset = (num == ohem_size_) ? 2 : 0; // if num == ohem_size_ when second pass

		calc_acc_gpu_kernel << <1, num >> >(
			num,
			feature_dim,
			avg_prob_->get(device_index)->gpu_data(),
			inputs_[1]->get(device_index)->gpu_data(),
			correct_->get(device_index)->mutable_gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data() + tag_offset,
			outputs_[0]->get(device_index)->mutable_gpu_data() + 1 + tag_offset
			);

#if _DEBUG
		//for (int i = 0; i < 4; i++)
		//	printf("i: %d, output: %.10f\n", i, outputs_[0]->get(device_index)->cpu_data()[i]);
#endif

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

			if (num == ohem_size_)
				outputs_[0]->get(device_index)->mutable_cpu_data()[5] = 1.f - (data_type)top_n_acc / num;
			else
				outputs_[0]->get(device_index)->mutable_cpu_data()[4] = 1.f - (data_type)top_n_acc / num;

		}

		// compute indicator
		if (force_random_) {
			std::vector<int> idx(num);
			for (int i = 0; i < num; i++)
				idx[i] = i;
			std::random_shuffle(idx.begin(), idx.end());

			/*for (int i = 0; i < num; i++)
			printf("i: %d, rank: %d\n", i, idx[i]);*/

			data_type *bp_indicator_data = bp_indicator_->get(device_index)->mutable_cpu_data();
			for (int i = 0; i < ohem_size_; i++)
				bp_indicator_data[idx[i]] = 1.f; // back prop

			for (int i = ohem_size_; i < num; i++)
				bp_indicator_data[idx[i]] = 0.f; // do not back prop
		}
		else {
			const data_type *avg_prob_data = avg_prob_->get(device_index)->cpu_data();
			const data_type *label_data = inputs_[1]->get(0)->cpu_data();

			std::vector<std::pair<data_type, int> > score_max(num);

			for (int i = 0; i < num; i++, avg_prob_data += feature_dim, label_data++) {

				int ground_truth = (int)floor(*label_data + 0.5f);
				data_type loss = -logf(avg_prob_data[ground_truth]);

				data_type s = 0.f;
				for (int k = 0; k < feature_dim; k++)
					s = fmaxf(s, avg_prob_data[k]);

				if (use_max_loss_)
					score_max[i] = std::make_pair(-loss, i); // smallest minus loss; max loss;
				else
					score_max[i] = std::make_pair(s, i); // smallest score

				//printf("i: %d, s: %.10f, loss: %.10f --- i: %d, s: %.10f\n", i, s, loss, score_max[i].second, score_max[i].first);
			}

			std::sort(score_max.begin(), score_max.end(), std::less<std::pair<data_type, int> >());

			/*for (int i = 0; i < num; i++)
				printf("rank: %d, i: %d, s: %.10f\n", i, score_max[i].second, score_max[i].first);*/

			data_type *bp_indicator_data = bp_indicator_->get(device_index)->mutable_cpu_data();
			for (int i = 0; i < ohem_size_; i++)
				bp_indicator_data[score_max[i].second] = 1.f; // smallest confidence; back prop

			for (int i = ohem_size_; i < num; i++)
				bp_indicator_data[score_max[i].second] = 0.f; // do not back prop

			/////
			//avg_prob_data = avg_prob_->get(device_index)->cpu_data();
			//for (int i = 0; i < num; i++, avg_prob_data += feature_dim) {
			//	data_type s = 0.f;
			//	for (int k = 0; k < feature_dim; k++)
			//		s = fmaxf(s, avg_prob_data[k]);
			//	printf("i: %d, s: %.10f --- indicator: %f\n", i, s, bp_indicator_data[i]);
			//}
		}
	}

	void softmax_ohem_layer::on_after_forward() {

		CHECK_EQ(bp_indicator_->size(), 1);
		CHECK_EQ(inputs_[1]->get(0)->channels(), 1);
		CHECK_EQ(inputs_[2]->size(), outputs_[2]->size());

		// select output data
		const int num = bp_indicator_->get(0)->num();
		const int item_size = inputs_[2]->get(0)->channels() * inputs_[2]->get(0)->height() * inputs_[2]->get(0)->width();

		const data_type *bp_indicator_data = bp_indicator_->get(0)->cpu_data();

		const data_type *input_label = inputs_[1]->get(0)->cpu_data();
		data_type *selected_label = outputs_[1]->get(0)->mutable_cpu_data();

		int idx_selected = 0;
		int idx_selected_label = 0;
		int device_index = 0;
		data_type* selected_data = outputs_[2]->get(device_index)->mutable_cpu_data();

		int idx_total = 0;
		for (int nd = 0; nd < inputs_[2]->size(); nd++)
		{
			const data_type* input_data = inputs_[2]->get(nd)->cpu_data();
			for (int i = 0; i < inputs_[2]->get(nd)->num(); i++, idx_total++) {

				//printf("nd: %d, i: %d, idx_total: %d; idx_selected: %d, idx_selected_label: %d, indicator: %f\n", nd, i, idx_total, idx_selected, idx_selected_label, bp_indicator_data[idx_total]);

				if (bp_indicator_data[idx_total] > 0) {
					memcpy(selected_data, input_data, item_size * sizeof(data_type));

					selected_label[idx_selected_label] = input_label[idx_total];
					idx_selected_label++;

					if (idx_selected + 1 < outputs_[2]->get(device_index)->num()) {
						selected_data += item_size;
						idx_selected++;
					}
					else {
						device_index++;
						if (device_index < outputs_[2]->size()) {
							idx_selected = 0;
							selected_data = outputs_[2]->get(device_index)->mutable_cpu_data();
						}
					}
				}
				input_data += item_size;
			} //i
		}//nd

#if _DEBUG
		///////////// for debug

		//////////////////////////////
		//{
		//	const int feature_dim = inputs_[0]->get(0)->dim_at(2);

		//	const data_type *avg_prob_data_tmp = avg_prob_->get(0)->cpu_data();
		//	const data_type *label_data = inputs_[1]->get(0)->cpu_data();

		//	data_type avg_loss = 0.f;

		//	const data_type *bp_indicator_data = bp_indicator_->get(0)->cpu_data();
		//	for (int i = 0; i < bp_indicator_->get(0)->num(); i++, avg_prob_data_tmp += feature_dim, label_data++) {

		//		int ground_truth = *label_data;
		//		data_type loss = -logf(avg_prob_data_tmp[ground_truth]);

		//		data_type s = 0.f;
		//		for (int k = 0; k < feature_dim; k++)
		//			s = fmaxf(s, avg_prob_data_tmp[k]);

		//		if (bp_indicator_data[i] > 0) {
		//			avg_loss += loss;
		//		}
		//		printf("i: %d, s: %.10f, loss: %.10f, bp_indicator_data: %.10f, ground_truth: %d\n", i, s, loss, bp_indicator_data[i], ground_truth);
		//	}//i
		//	printf("\n");

		//	avg_loss /= ohem_size_;
		//	printf("OHEM loss: %.10f\n", avg_loss);
		//}
#endif
	}

	__global__ void static bp_softlost_gpu_kernel(const int n, const int feature_dim, const int spatial_size,
		const data_type *probs, const data_type *label, const data_type *bp_indicator,
		data_type *diff, const data_type coeff, const data_type scale_targets,
		const int total_ins) {

		CUDA_KERNEL_LOOP(index, n) {
			int dest_idx = index;

			index /= spatial_size;
			int cur_ins = index / feature_dim;
			int cur_dim = index % feature_dim;
			//int total_ins = n / feature_dim / spatial_size;

			int ground_truth = (label[cur_ins] >= 0 ? (int)(label[cur_ins] + 0.5) : -1);

			if (ground_truth >= 0) {
				if (ground_truth == cur_dim) {

					//data_type v = coeff * (probs[dest_idx] - 1) / (data_type)total_ins / (data_type)spatial_size;
					data_type v = bp_indicator[cur_ins] * coeff * (probs[dest_idx] - 1) / (data_type)total_ins / (data_type)spatial_size;

					if (scale_targets == 0) {
						diff[dest_idx] = v;
					}
					else {
						diff[dest_idx] = diff[dest_idx] * scale_targets + v;
					}
				}
				else {

					//data_type v = coeff * probs[dest_idx] / (data_type)total_ins / (data_type)spatial_size;
					data_type v = bp_indicator[cur_ins] * coeff * probs[dest_idx] / (data_type)total_ins / (data_type)spatial_size;

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

	void softmax_ohem_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
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
				bp_indicator_->get(device_index)->gpu_data(),
				inputs_[0]->get(device_index)->mutable_gpu_diff(),
				coeff_,
				beta_acts,
				ohem_size_
				);
		}
	}
}