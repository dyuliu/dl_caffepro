
#include <caffepro/layers/cluster_loss_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/utils/filler.h>

#include <boost/scoped_ptr.hpp>

namespace caffepro {
	cluster_loss_layer::cluster_loss_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);

		attr_.usage = layer_attribute::USAGE_LOSS;
	}

	cluster_loss_layer::~cluster_loss_layer() {
		release_all();
	}

	void cluster_loss_layer::init() {
		check_input();

		reset_centers_ = false;

		coeff_ = (data_type)1.f;
		if (layer_param_.has_loss_param() && layer_param_.loss_param().has_coeff()) {
			coeff_ = layer_param_.loss_param().coeff();
		}
		
		// same channels
		CHECK(inputs_[0]->same_dim_at(2));

		num_centers_ = layer_param_.cluster_param().num_centers();
		num_dims_ = inputs_[0]->dim_at(2);

		// init weights
		boost::scoped_ptr<device_blob> weight_template(
			device_blob::create_4d(context_, num_centers_, num_dims_, 1, 1));
		weights_.resize(1);
		weights_[0].reset(new node_blob());
		weights_[0]->add_like(context_, *weight_template, *inputs_[0]);

		if (layer_param_.cluster_param().weight_filler().type() != "reset") {
			boost::scoped_ptr<filler> weight_filler(filler::create(context_, layer_param_.cluster_param().weight_filler()));
			weight_filler->fill(*weights_[0]->get(0));
			weights_[0]->broadcast_data_via_gpu(0);
		}
		else {
			reset_centers_ = true;
		}

		// init outputs
		int n_devices = (int)inputs_[0]->size();
		outputs_[0]->tags() = std::vector<std::string>({
			"Loss",
			"Diversity"
		});

		for (int nd = 0; nd < n_devices; nd++) {
			outputs_[0]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
				context_, 1, (int)(outputs_[0]->tags().size()), 1, 1, inputs_[0]->get(nd)->device_id())));
		}

		// init internal structures
		distance_matrix_.reset(new node_blob());
		assign_matrix_.reset(new node_blob());
		assign_matrix_back_.reset(new node_blob());
		loss_matrix_.reset(new node_blob());

		for (int nd = 0; nd < n_devices; nd++) {
			assign_matrix_back_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
				context_, num_centers_, 1, 1, 1, inputs_[0]->get(nd)->device_id()
				)));
		}
	}

	void cluster_loss_layer::resize() {
		check_input();

		CHECK(inputs_[0]->dim_at(2) == num_dims_);
		bool init = (distance_matrix_->size() == 0);

		int n_device = (int)inputs_[0]->size();
		for (int nd = 0; nd < n_device; nd++) {
			if (init || inputs_[0]->reshaped()) {
				distance_matrix_->set_4d(
					nd,
					inputs_[0]->get(nd)->num(),
					num_centers_,
					inputs_[0]->get(nd)->height(),
					inputs_[0]->get(nd)->width(),
					inputs_[0]->get(nd)->device_id(),
					context_
					);

				assign_matrix_->set_4d(
					nd,
					inputs_[0]->get(nd)->num(),
					1,
					inputs_[0]->get(nd)->height(),
					inputs_[0]->get(nd)->width(),
					inputs_[0]->get(nd)->device_id(),
					context_
					);

				loss_matrix_->set_4d(
					nd,
					inputs_[0]->get(nd)->num(),
					1,
					inputs_[0]->get(nd)->height(),
					inputs_[0]->get(nd)->width(),
					inputs_[0]->get(nd)->device_id(),
					context_
					);
			}
		}
	}

	__global__ static void calc_distance_matrix(const int count, const int spatial_size,
		const int num_centers, const int num_dims, const data_type *inputs, const data_type *clusters,
		data_type *distance_matrix) {
		CUDA_KERNEL_LOOP(index, count) {
			const int hw = index % spatial_size;
			const int c = index / spatial_size % num_centers;
			const int n = index / spatial_size / num_centers;

			inputs += n * num_dims * spatial_size + hw;
			clusters += c * num_dims;

			data_type v = 0.f;
			for (int i = 0; i < num_dims; i++) {
				data_type vi = inputs[i * spatial_size];
				data_type vc = clusters[i];
				//v += fabsf(vi - vc); // l1
				v += (vi - vc) * (vi - vc); // l2
			}

			distance_matrix[index] = v;
		}
	}

	__global__ static void calc_assign_matrix(const int count, const int spatial_size,
		const int num_centers, const data_type *distance_matrix,
		data_type *assign_matrix, data_type *loss_matrix) {
		CUDA_KERNEL_LOOP(index, count) {
			const int hw = index % spatial_size;
			const int n = index / spatial_size;

			distance_matrix += n * num_centers * spatial_size + hw;
			
			data_type min_dis = FLT_MAX;
			int ass_center = -1;
			for (int i = 0; i < num_centers; i++) {
				data_type v = distance_matrix[i * spatial_size];
				if (v < min_dis) {
					min_dis = v;
					ass_center = i;
				}
			}

			assign_matrix[index] = (data_type)ass_center;
			loss_matrix[index] = min_dis;
		}
	}

	__global__ static void calc_assign_matrix_back(const int count, const int spatial_size, const int input_num, const int num_centers,
		const data_type *distance_matrix, data_type *assign_matrix_back) {
		CUDA_KERNEL_LOOP(index, count) {
			data_type min_dis = FLT_MAX;
			int ass_back_spatial = -1, ass_back_inputnum = -1;

			for (int n = 0; n < input_num; n++) {
				for (int s = 0; s < spatial_size; s++) {
					data_type v = distance_matrix[(n * num_centers + index) * spatial_size + s];

					if (v < min_dis) {
						min_dis = v;
						ass_back_spatial = s;
						ass_back_inputnum = n;
					}
				}
			}

			assign_matrix_back[index] = ass_back_inputnum * spatial_size + ass_back_spatial;
		}
	}

	void cluster_loss_layer::on_before_forward() {
		if (reset_centers_) {
			cublas_wrapper<data_type> cublas(context_, inputs_[0]->get(0)->device_id());
			int num = inputs_[0]->get(0)->num();
			int spatial_size = inputs_[0]->get(0)->height() * inputs_[0]->get(0)->width();
			weights_[0]->get(0)->fill_data(0.f);
			for (int i = 0; i < num_centers_; i++) {
				int sel_num = (int)(num * (double)rand() / (RAND_MAX + 1));
				int sel_spatial = (int)(spatial_size * (double)rand() / (RAND_MAX + 1));

				const data_type *src = inputs_[0]->get(0)->gpu_data() + inputs_[0]->get(0)->offset(sel_num) + sel_spatial;
				cublas.axpby(
					num_dims_,
					(data_type)1.f,
					src,
					spatial_size,
					(data_type)0.f,
					weights_[0]->get(0)->mutable_gpu_data() + weights_[0]->get(0)->offset(i),
					1
					);
			}

			weights_[0]->broadcast_data_via_gpu(0);
			context_->sync_all_devices();
			reset_centers_ = false;
		}
	}

	void cluster_loss_layer::on_forward(int device_index) {
		auto &input = *inputs_[0]->get(device_index);

		int count_dis = distance_matrix_->get(device_index)->count();
		KERNEL_CALL(calc_distance_matrix, count_dis)(
			count_dis,
			input.height() * input.width(),
			num_centers_,
			num_dims_,
			input.gpu_data(),
			weights_[0]->get(device_index)->gpu_data(),
			distance_matrix_->get(device_index)->mutable_gpu_data()
			);

		int count_ass = assign_matrix_->get(device_index)->count();
		KERNEL_CALL(calc_assign_matrix, count_ass)(
			count_ass,
			input.height() * input.width(),
			num_centers_,
			distance_matrix_->get(device_index)->gpu_data(),
			assign_matrix_->get(device_index)->mutable_gpu_data(),
			loss_matrix_->get(device_index)->mutable_gpu_data()
			);

		KERNEL_CALL(calc_assign_matrix_back, num_centers_)(
			num_centers_,
			input.height() * input.width(),
			input.num(),
			num_centers_,
			distance_matrix_->get(device_index)->gpu_data(),
			assign_matrix_back_->get(device_index)->mutable_gpu_data()
			);

		outputs_[0]->get(device_index)->mutable_cpu_data()[0] = (data_type)loss_matrix_->get(device_index)->mean() / num_dims_;
	}

	void cluster_loss_layer::on_after_forward() {
		// merge loss
		int n_devices = (int)outputs_[0]->size();
		data_type *result_data = outputs_[0]->get(0)->mutable_cpu_data();

		for (int nd = 1; nd < n_devices; nd++) {
			const data_type *dev_result_data = outputs_[0]->get(nd)->cpu_data();
			result_data[0] += dev_result_data[0];
		}

		result_data[0] /= n_devices;

		// merge diversity
		std::set<int> unique_assign;
		for (int nd = 0; nd < n_devices; nd++) {
			const data_type *dev_assign = assign_matrix_->get(nd)->cpu_data();
			int count = assign_matrix_->get(nd)->count();
			for (int i = 0; i < count; i++) {
				int v = (int)(dev_assign[i] + 0.5f);
				unique_assign.insert(v);
			}
		}

		result_data[1] = (data_type)unique_assign.size();
	}

	__global__ static void bp_acts_kernel(const int count, const int spatial_size, 
		const int num_dims, const data_type *inputs, const data_type *clusters, const data_type *assign_matrix,
		data_type *act_diff, const data_type coeff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, count) {
			const int hw = index % spatial_size;
			const int c = index / spatial_size % num_dims;
			const int n = index / spatial_size / num_dims;

			int idx_center = (int)(assign_matrix[n * spatial_size + hw] + 0.5f);
			data_type vc = clusters[idx_center * num_dims + c];
			data_type vi = inputs[index];
			//data_type diff = coeff * (vi >= vc ? 1 : -1); // l1
			data_type diff = coeff * (vi - vc) * 2; // l2
			if (scale_targets == 0) {
				act_diff[index] = diff;
			}
			else {
				act_diff[index] = act_diff[index] * scale_targets + diff;
			}
		}
	}

	__global__ static void bp_weights_kernel(const int count, const int spatial_size, const int num,
		const int num_dims, const data_type *inputs, const data_type *clusters, const data_type *assign_matrix, const data_type *assign_matrix_back,
		data_type *cluster_diff, const data_type coeff, const data_type scale_targets) {
		CUDA_KERNEL_LOOP(index, count) {
			const int c = index % num_dims;
			const int idx_center = index / num_dims;

			data_type diff = 0.f;
			int k = 0;
			bool assigned = false;
			for (int n = 0; n < num; n++) {
				for (int s = 0; s < spatial_size; s++, k++) {
					int ass_index = (int)(assign_matrix[k] + 0.5f);

					if (ass_index == idx_center) {
						data_type vi = inputs[(n * num_dims + c) * spatial_size + s];
						data_type vc = clusters[index];
						//diff += vc > vi ? 1 : -1; // l1
						diff += (vc - vi) * 2; // l2
						assigned = true;
					}
				}
			}

			if (!assigned) {
				int nearest_input_pos = assign_matrix_back[idx_center];
				int nearest_input_index = (nearest_input_pos / spatial_size * num_dims + c) * spatial_size + nearest_input_pos % spatial_size;

				data_type vi = inputs[nearest_input_index];
				data_type vc = clusters[index];

				diff = (vc - vi) * 2; // l2
			}

			diff *= coeff;

			if (scale_targets == 0) {
				cluster_diff[index] = diff;
			}
			else {
				cluster_diff[index] = cluster_diff[index] * scale_targets + diff;
			}
		}
	}

	void cluster_loss_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		auto &inputs = *inputs_[0]->get(device_index);
		auto &clusters = *weights_[0]->get(device_index);
		auto &assign_matrix = *assign_matrix_->get(device_index);
		auto &assign_matrix_back = *assign_matrix_back_->get(device_index);

		if (should_bp(bp_acts, 0)) {
			const data_type beta_acts = get_beta(clear_acts_diff, 0);

			int count = inputs.count();
			KERNEL_CALL(bp_acts_kernel, count)(
				count,
				inputs.height() * inputs.width(),
				num_dims_,
				inputs.gpu_data(),
				clusters.gpu_data(),
				assign_matrix.gpu_data(),
				inputs.mutable_gpu_diff(),
				coeff_ / (inputs.count()),
				beta_acts
				);
		}

		if (should_bp(bp_weights, 0)) {
			const data_type beta_weights = get_beta(clear_weights_diff, 0);

			int count = clusters.count();
			KERNEL_CALL(bp_weights_kernel, count)(
				count,
				inputs.height() * inputs.width(),
				inputs.num(),
				num_dims_,
				inputs.gpu_data(),
				clusters.gpu_data(),
				assign_matrix.gpu_data(),
				assign_matrix_back.gpu_data(),
				clusters.mutable_gpu_diff(),
				coeff_ / (inputs.count()),
				beta_weights
				);
		}
	}
}