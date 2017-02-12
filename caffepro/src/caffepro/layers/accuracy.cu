
#include <caffepro/layers/accuracy_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/context/common_names.h>

#define MAX_MINIBATCH 512

namespace caffepro {
	accuracy_layer::accuracy_layer(caffepro_context *context, const LayerParameter &param)
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

	accuracy_layer::~accuracy_layer() {
		release_all();
	}

	void accuracy_layer::init() {
		check_input();
		output_top_n_ = 0;

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
	}

	void accuracy_layer::resize() {
		check_input();

		CHECK(inputs_[1]->get(0)->count() == inputs_[0]->get(0)->num());
		CHECK_EQ(inputs_[0]->get(0)->height(), 1);
		CHECK_EQ(inputs_[0]->get(0)->width(), 1);
	}

	__global__ static void calc_acc_gpu_kernel(const int n, const int dim, const data_type *probs,
		const data_type *label, data_type *acc, data_type *loss) {
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

	void accuracy_layer::on_forward(int device_index) {
		const int num = inputs_[0]->get(device_index)->num();
		const int feature_dim = inputs_[0]->get(device_index)->dim_at(2);

		CHECK_LE(num, MAX_MINIBATCH);

		calc_acc_gpu_kernel<<<1, num >>>(
			num,
			feature_dim,
			inputs_[0]->get(device_index)->gpu_data(),
			inputs_[1]->get(device_index)->gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data() + 1
			);

		// optional: calc top_n error
		// cpu code
		if (output_top_n_ >= 1) {
			const data_type *avg_prob_data = inputs_[0]->get(device_index)->cpu_data();
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

	void accuracy_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		NOT_IMPLEMENTED;
	}
}