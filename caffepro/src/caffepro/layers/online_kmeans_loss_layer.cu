
#include <caffepro/layers/online_kmeans_loss_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/utils/filler.h>

#include <boost/scoped_ptr.hpp>

namespace caffepro {
	online_kmeans_loss_layer::online_kmeans_loss_layer(caffepro_context *context, const LayerParameter &param)
		: cluster_loss_layer(context, param) {
	}

	void online_kmeans_loss_layer::init() {
		cluster_loss_layer::init();

		// clear weight diff
		weights_[0]->get(0)->fill_diff(0.f);
		weights_[0]->broadcast_diff_via_gpu(0);

		update_interval_ = layer_param_.online_kmeans_param().update_interval();
		update_iters_ = layer_param_.online_kmeans_param().update_iters();
		current_iter_ = 0;
		current_kmeans_batch_ = 0;

		CHECK_GE(update_interval_, update_iters_);
		CHECK_GT(update_iters_, 0);

		prepare_centers_.reset(new node_blob());
		prepare_centers_->add_like(*weights_[0]); // create on default device
		center_count_.resize(num_centers_);
		std::fill(center_count_.begin(), center_count_.end(), 1);
		
		prepare_distance_matrix_.reset(new node_blob());
		prepare_assign_matrix_.reset(new node_blob());
	}

	void online_kmeans_loss_layer::resize() {
		cluster_loss_layer::resize();

		int n_devices = (int)inputs_[0]->size();
		if (prepare_distance_matrix_->size() == 0) {
			prepare_distance_matrix_->add_like(*distance_matrix_);
			prepare_assign_matrix_->add_like(*assign_matrix_);
		}
		else {
			for (int nd = 0; nd < n_devices; nd++) {
				if (inputs_[0]->get(nd)->reshaped()) {
					prepare_distance_matrix_->get(nd)->reshape_like(*distance_matrix_->get(nd));
					prepare_assign_matrix_->get(nd)->reshape_like(*assign_matrix_->get(nd));
				}
			}
		}
	}

	void online_kmeans_loss_layer::on_forward(int device_index) {
		cluster_loss_layer::on_forward(device_index);
	}

	void online_kmeans_loss_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		// do not bp weights
		cluster_loss_layer::on_backward(device_index, bp_acts, 0, clear_acts_diff, 0);
	}

	void online_kmeans_loss_layer::on_after_backward() {
		cluster_loss_layer::on_after_backward();

		if (current_kmeans_batch_ == 0) {
			if (current_iter_ % update_interval_ == 0) { // start online kmeans
				init_centers();

				current_kmeans_batch_ = 1;
			}
		}
		else {
			current_kmeans_batch_++;
		}

		current_iter_++;

		int n_devices = (int)inputs_[0]->size();
		if (current_kmeans_batch_ > 0) {
			// calc assign
			for (int nd = 0; nd < n_devices; nd++) {
				find_nearest(nd);
			}

			// do kmeans
			for (int nd = 0; nd < n_devices; nd++) {
				minibatch_kmeans(nd);
			}
			
			// since we only updated centers on the first device
			// we need to broadcast it to all devices
			prepare_centers_->broadcast_data_via_gpu(0);

			if (current_kmeans_batch_ >= update_iters_) {
				// finish update; apply centers

				for (int nd = 0; nd < n_devices; nd++) {
					weights_[0]->get(nd)->copy_data_from_via_gpu(*prepare_centers_->get(nd));
				}

				LOG(INFO) << "KMeans centers updated";

				current_kmeans_batch_ = 0;
			}
		}
	}

	void online_kmeans_loss_layer::init_centers() {
		prepare_centers_->get(0)->copy_data_from_via_gpu(*weights_[0]->get(0));
		data_type *center_data = prepare_centers_->get(0)->mutable_cpu_data();

		int n_devices = (int)inputs_[0]->size();
		for (int c = 0; c < num_centers_; c++) {
			if (center_count_[c] == 0) {
				int sel_dev = rand() % n_devices;
				auto &input = *inputs_[0]->get(sel_dev);
				int nearest_code = (int)(assign_matrix_back_->get(sel_dev)->cpu_data()[c] + 0.5f);
				int spatial_size = input.height() * input.width();
				int nearest_n = nearest_code / spatial_size, nearest_hw = nearest_code % spatial_size;
				const data_type *src_data = input.cpu_data() + nearest_n * input.channels() * spatial_size + nearest_hw;
				data_type *dest_data = center_data + c * num_dims_;
				for (int i = 0; i < num_dims_; i++) {
					dest_data[i] = src_data[i * spatial_size];
				}
			}
		}

		prepare_centers_->broadcast_data_via_gpu(0);
		std::fill(center_count_.begin(), center_count_.end(), 0);
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
		data_type *assign_matrix) {
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
		}
	}

	void online_kmeans_loss_layer::find_nearest(int device_index) {
		ENTER_DEVICE_CONTEXT(inputs_[0]->get(device_index)->device_id())
			auto &input = *inputs_[0]->get(device_index);

			int count_dis = prepare_distance_matrix_->get(device_index)->count();
			KERNEL_CALL(calc_distance_matrix, count_dis)(
				count_dis,
				input.height() * input.width(),
				num_centers_,
				num_dims_,
				input.gpu_data(),
				prepare_centers_->get(device_index)->gpu_data(),
				prepare_distance_matrix_->get(device_index)->mutable_gpu_data()
				);

			int count_ass = prepare_assign_matrix_->get(device_index)->count();
			KERNEL_CALL(calc_assign_matrix, count_ass)(
				count_ass,
				input.height() * input.width(),
				num_centers_,
				prepare_distance_matrix_->get(device_index)->gpu_data(),
				prepare_assign_matrix_->get(device_index)->mutable_gpu_data()
				);
		EXIT_DEVICE_CONTEXT;
	}

	void online_kmeans_loss_layer::minibatch_kmeans(int device_index) {
		// always update centers on the first device
		data_type *centers = prepare_centers_->get(0)->mutable_cpu_data();

		const data_type *assigns = prepare_assign_matrix_->get(device_index)->cpu_data();

		auto &input = *inputs_[0]->get(device_index);
		int spatial_size = input.height() * input.width();
		int num = input.num();
		
		const data_type *input_data = input.cpu_data();

		for (int n = 0; n < num; n++) {
			for (int s = 0; s < spatial_size; s++) {
				int center_index = (int)(assigns[n * spatial_size + s] + 0.5f);
				center_count_[center_index]++;

				data_type ratio = 1.f / center_count_[center_index];
				const data_type *src_data = input_data + n * num_dims_ * spatial_size + s;
				data_type *dest_data = centers + center_index * num_dims_;
				for (int i = 0; i < num_dims_; i++) {
					dest_data[i] = dest_data[i] * (1 - ratio) + src_data[i * spatial_size] * ratio;
				}
			}
		}
	}
}