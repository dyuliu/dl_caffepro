
#include <caffepro/layers/box_regression_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/utils.h>

#define MAX_MINIBATCH 512

namespace caffepro {
	box_regression_layer::box_regression_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 3;
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

	box_regression_layer::~box_regression_layer() {
		release_all();
	}

	void box_regression_layer::init() {
		check_input();

		coeff_ = (data_type)1.f;
		if (layer_param_.has_loss_param() && layer_param_.loss_param().has_coeff()) {
			coeff_ = layer_param_.loss_param().coeff();
		}
	}

	void box_regression_layer::resize() {
		check_input();

		CHECK(inputs_[0]->same_dim_at(2));

		int n_devices = (int)inputs_[0]->size();

		if (outputs_[0]->size() == 0) {
			outputs_[0]->tags().resize(4);
			outputs_[0]->tags()[0] = "Loss";
			outputs_[0]->tags()[1] = "IoU before Regress";
			outputs_[0]->tags()[2] = "IoU after Regress";
			outputs_[0]->tags()[3] = "Regress Boxes";

			for (int nd = 0; nd < n_devices; nd++) {
				CHECK_EQ(inputs_[0]->get(nd)->inner_count() % 4, 0);
				CHECK_EQ(inputs_[1]->get(nd)->inner_count(), 4);
				CHECK_EQ(inputs_[2]->get(nd)->inner_count(), 1);

				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, 1, (int)outputs_[0]->tags().size(), 1, 1, inputs_[0]->get(nd)->device_id())
					));
			}
		}
		else {
			for (int nd = 0; nd < n_devices; nd++) {
				CHECK_EQ(inputs_[0]->get(nd)->inner_count() % 4, 0);
				CHECK_EQ(inputs_[1]->get(nd)->inner_count(), 4);
				CHECK_EQ(inputs_[2]->get(nd)->inner_count(), 1);
			}
		}
	}

	__device__ static bool box_valid(const data_type *box) {
		return box[0] <= box[2] && box[1] <= box[3];
	}

	__device__ static data_type area(const data_type *box) {
		if (box_valid(box)) {
			return (box[2] - box[0]) * (box[3] - box[1]);
		}
		else {
			return 0.f;
		}
	}

	__device__ static data_type calc_iou(const data_type *box_a, const data_type *box_b) {
		if (box_valid(box_a) && box_valid(box_b)) {
			data_type inc_box[4] = {
				fmax(box_a[0], box_b[0]),
				fmax(box_a[1], box_b[1]),
				fmin(box_a[2], box_b[2]),
				fmin(box_a[3], box_b[3])
			};
			data_type area_inc = area(inc_box);
			data_type area_a = area(box_a), area_b = area(box_b);
			return area_inc / (area_a + area_b - area_inc);
		}
		else {
			return 0.f;
		}
	}

	__global__ static void box_regress_forward(const int dim, const data_type *predict_offset,
		const data_type *gt_offset, const data_type *label, data_type *stat) {

		int cur_ins = threadIdx.x;
		predict_offset += cur_ins * dim * 4;
		gt_offset += cur_ins * 4;

		int gt_label = (int)(label[cur_ins] + 0.5f);

		if (gt_label < dim) {
			predict_offset += gt_label * 4;

			// calc loss 
			data_type loss = 0.f;
			for (int i = 0; i < 4; i++) {
				data_type diff = fabs(predict_offset[i] - gt_offset[i]);
				if (diff >= 1.f) loss += diff;
				else loss += diff * diff;
			}

			data_type ref_box[4] = { 0.f, 0.f, 1.f, 1.f };
			data_type predict_box[4] = { predict_offset[0], predict_offset[1], predict_offset[2] + 1.f, predict_offset[3] + 1.f };
			data_type gt_box[4] = { gt_offset[0], gt_offset[1], gt_offset[2] + 1.f, gt_offset[3] + 1.f };

			data_type iou_before = calc_iou(ref_box, gt_box);
			data_type iou_after = calc_iou(predict_box, gt_box);

			atomicAdd(stat, loss);
			atomicAdd(stat + 1, iou_before);
			atomicAdd(stat + 2, iou_after);
			atomicAdd(stat + 3, 1.f);
		}
	}

	void box_regression_layer::on_forward(int device_index) {
		auto &predict_offset = *inputs_[0]->get(device_index);
		auto &gt_offset = *inputs_[1]->get(device_index);
		auto &label = *inputs_[2]->get(device_index);

		int num = predict_offset.num();
		int dim = predict_offset.inner_count() / 4;
		outputs_[0]->get(device_index)->fill_data(0.f);

		box_regress_forward<<<1, num>>>(
			dim,
			predict_offset.gpu_data(),
			gt_offset.gpu_data(),
			label.gpu_data(),
			outputs_[0]->get(device_index)->mutable_gpu_data()
			);
	}

	void box_regression_layer::on_after_forward() {
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

		int num = (int)(result_data[3] + 0.5f);
		if (num > 0) {
			result_data[0] /= num;
			result_data[1] /= num;
			result_data[2] /= num;
		}
	}

	__global__ static void box_regress_backward(const int dim, const data_type *predict_offset,
		const data_type *gt_offset, const data_type *label, data_type *predict_diff, 
		const data_type coeff, const data_type scale_targets) {

		int cur_ins = threadIdx.x;
		predict_offset += cur_ins * dim * 4;
		predict_diff += cur_ins * dim * 4;
		gt_offset += cur_ins * 4;

		int gt_label = (int)(label[cur_ins] + 0.5f);

		if (gt_label < dim) {
			predict_offset += gt_label * 4;
			predict_diff += gt_label * 4;

			for (int i = 0; i < 4; i++) {
				data_type diff = predict_offset[i] - gt_offset[i];
				data_type v = 0;
				if (diff >= 1.f) v = 1.f;
				else if (diff <= -1.f) v = -1.f;
				else v = diff;

				if (scale_targets == 0) predict_diff[i] = v * coeff;
				else predict_diff[i] = predict_diff[i] * scale_targets + v * coeff;
			}
		}
	}

	void box_regression_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			auto &predict_offset = *inputs_[0]->get(device_index);
			auto &gt_offset = *inputs_[1]->get(device_index);
			auto &label = *inputs_[2]->get(device_index);

			if (beta_acts == 0) {
				predict_offset.fill_diff(0.f);
			}

			int num = predict_offset.num();
			int dim = predict_offset.inner_count() / 4;

			box_regress_backward<<<1, num>>>(
				dim,
				predict_offset.gpu_data(),
				gt_offset.gpu_data(),
				label.gpu_data(),
				predict_offset.mutable_gpu_diff(),
				coeff_ / num,
				beta_acts
				);
		}
	}
}