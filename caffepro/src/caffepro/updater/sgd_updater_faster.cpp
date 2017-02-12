
#include <caffepro/updater/sgd_updater_faster.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	using std::vector;

	sgd_updater_faster::sgd_updater_faster(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics)
		: sgd_updater(context, weight_info, param, update_metrics) {
		init();
	}

	sgd_updater_faster::~sgd_updater_faster() {
		// nothing to do
	}

	struct weight_info_stat {
		data_type local_lr;
		data_type local_wc;
		long long encoded_devices;
		int index;

		bool operator < (const weight_info_stat &other) {
			return local_lr < other.local_lr
				|| (local_lr == other.local_lr && local_wc < other.local_wc)
				|| (local_lr == other.local_lr && local_wc == other.local_wc && encoded_devices < other.encoded_devices);
		}

		bool operator == (const weight_info_stat &other) {
			return local_lr == other.local_lr && local_wc == other.local_wc && encoded_devices == other.encoded_devices;
		}
	};

	void sgd_updater_faster::init() {
		// clustering weights
		int total_gpus;
		CUDA_CHECK(cudaGetDeviceCount(&total_gpus));

		vector<weight_info_stat> wstat;
		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			auto &blob = *weight_info_.weights[i];
			long long encode = (int)blob.size();
			for (int nd = 0; nd < (int)blob.size(); nd++) {
				encode = encode * total_gpus + blob.get(nd)->device_id();
			}
			
			weight_info_stat stat = {
				weight_info_.learning_rate[i],
				weight_info_.weight_decay[i],
				encode,
				i
			};

			wstat.push_back(stat);
		}

		std::sort(wstat.begin(), wstat.end());
		for (int i = 0; i < (int)wstat.size(); i++) {
			if (i > 0 && wstat[i] == wstat[i - 1]) {
				update_groups_.back().member_indexes.push_back(wstat[i].index);
			}
			else {
				update_group ug;
				ug.local_lr = wstat[i].local_lr;
				ug.local_wc = wstat[i].local_wc;
				ug.member_indexes.push_back(wstat[i].index);
				update_groups_.push_back(ug);
			}
		}

		// init group data
		for (update_group &ug : update_groups_) {
			int count = 0;
			for (int member_index : ug.member_indexes) {
				count += weight_info_.weights[member_index]->get(0)->count();
			}

			ug.group_data.reset(new node_blob());
			node_blob &ref_blob = *weight_info_.weights[ug.member_indexes.front()];
			for (int nd = 0; nd < (int)ref_blob.size(); nd++) {
				ug.group_data->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
					context_, count, 1, 1, 1, ref_blob.get(nd)->device_id()
					)));
			}
			ug.group_history.reset(device_blob::create_4d(
				context_, count, 1, 1, 1, ref_blob.get(0)->device_id()
				));

			// replace dev memory
			int offset = 0;
			for (int member_index : ug.member_indexes) {
				// weight memory
				node_blob &blob = *weight_info_.weights[member_index];
				for (int nd = 0; nd < (int)blob.size(); nd++) {
					CHECK_EQ(blob.get(nd)->device_id(), ug.group_data->get(nd)->device_id());

					blob.get(nd)->data_storage()->use_external_dev_memory(
						ug.group_data->get(nd)->mutable_gpu_data() + offset,
						blob.get(nd)->count() * sizeof(data_type)
						);
					blob.get(nd)->diff_storage()->use_external_dev_memory(
						ug.group_data->get(nd)->mutable_gpu_diff() + offset,
						blob.get(nd)->count() * sizeof(data_type)
						);
				}

				// history memory
				device_blob &his = *history_[member_index];
				CHECK_EQ(his.device_id(), ug.group_history->device_id());
				CHECK_EQ(his.count(), blob.get(0)->count());
				his.data_storage()->use_external_dev_memory(
					ug.group_history->mutable_gpu_data() + offset,
					his.count() * sizeof(data_type)
					);

				offset += blob.get(0)->count();
			}
		}

		// clear memory fragment
		for (int nd = 0; nd < total_gpus; nd++) {
			context_->get_device(nd)->memory()->gc();
		}
	}

	void sgd_updater_faster::update(data_type global_lr, data_type global_wc, bool ignore_zero_lr) {
		// group update
		if (!ignore_zero_lr || global_lr != 0) {
			// move weight and history heads to gpu (very important to break the "sync" state)
			for (auto &weight : weight_info_.weights) {
				for (int nd = 0; nd < (int)weight->size(); nd++) {
					weight->get(nd)->mutable_gpu_data();
					weight->get(nd)->mutable_gpu_diff();
				}
			}
			for (auto &his : history_) {
				his->mutable_gpu_data(); 
				// no need to process gpu_diff for history
			}
			//clock_t clk1 = clock();
			merge_diff(ignore_zero_lr);
			//clock_t clk2 = clock();
			for (int i = 0; i < (int)update_groups_.size(); i++) {
				data_type lr = update_groups_[i].local_lr * global_lr;
				data_type wc = update_groups_[i].local_wc * global_wc;

				if (!ignore_zero_lr || lr != 0) {
					on_update_first_device(i, lr, wc);
				}
			}
			context_->sync_all_devices();
			//clock_t clk3 = clock();
			broadcast_data(ignore_zero_lr);

			//clock_t clk4 = clock();
			//LOG(INFO) << clk2 - clk1;
			//LOG(INFO) << clk3 - clk2;
			//LOG(INFO) << clk4 - clk3;
		}
	}

	void sgd_updater_faster::merge_diff(bool ignore_zero_lr) {
		int max_devices = 0;
		for (int group_id = 0; group_id < (int)update_groups_.size(); group_id++) {
			if (!ignore_zero_lr || update_groups_[group_id].local_lr != 0) {
				max_devices = std::max(max_devices, (int)update_groups_[group_id].group_data->size());
			}
		}

		// binary merge
		for (int step = 1; step < max_devices; step *= 2) {
			for (int group_id = 0; group_id < (int)update_groups_.size(); group_id++) {
				if (!ignore_zero_lr || update_groups_[group_id].local_lr != 0) {
					auto &node = *update_groups_[group_id].group_data;
					int n_devices = (int)node.size();

					for (int nd = 0; nd + step < n_devices; nd += step * 2) {
						int dest_device_id = node[nd]->device_id();
						cublas_wrapper<data_type> cublas(context_, dest_device_id);
						data_type *buffer = reinterpret_cast<data_type *>(
							context_->get_device(dest_device_id)->memory()->allocate(node[nd]->count() * sizeof(data_type))
							);
						cublas.copy(node[nd]->count(), node[nd + step]->gpu_diff(), buffer);
						cublas.axpy(node[nd]->count(), (data_type)1.f, buffer, node[nd]->mutable_gpu_diff());
						context_->get_device(dest_device_id)->memory()->free(buffer);
					}
				}
			}
			context_->sync_all_devices();
		}
	}

	void sgd_updater_faster::broadcast_data(bool ignore_zero_lr) {
		//for (int group_id = 0; group_id < (int)update_groups_.size(); group_id++) {
		//	if (!ignore_zero_lr || update_groups_[group_id].local_lr != 0) {
		//		update_groups_[group_id].group_data->broadcast_data_via_gpu(0);
		//	}
		//}

		int max_devices = 0;
		for (int group_id = 0; group_id < (int)update_groups_.size(); group_id++) {
			if (!ignore_zero_lr || update_groups_[group_id].local_lr != 0) {
				max_devices = std::max(max_devices, (int)update_groups_[group_id].group_data->size());
			}
		}
		
		// binary broadcast
		for (int step = 1; step < max_devices; step *= 2) {
			for (int group_id = 0; group_id < (int)update_groups_.size(); group_id++) {
				if (!ignore_zero_lr || update_groups_[group_id].local_lr != 0) {
					//update_groups_[group_id].group_data->broadcast_data_via_gpu(0);
					
					auto &node = *update_groups_[group_id].group_data;
					int n_devices = (int)node.size();
					for (int nd = step; nd < step + step && nd < n_devices; nd++) {
						int dest_device_id = node[nd]->device_id();
						cublas_wrapper<data_type> cublas(context_, dest_device_id);
						cublas.copy(node[nd]->count(), node[nd - step]->gpu_data(), node[nd]->mutable_gpu_data());
					}
				}
			}
			context_->sync_all_devices();
		}
	}

	void sgd_updater_faster::on_update_first_device(int param_id, data_type lr, data_type wc) {
		device_blob &weight = *update_groups_[param_id].group_data->get(0);
		device_blob &his = *update_groups_[param_id].group_history;
		cublas_wrapper<data_type> cublas(context_, weight.device_id());

		//LOG(ERROR) << weight.mean(false) << " " << weight.variance(false) << " " << weight.mean(true) << " " << weight.variance(true);

		cublas.axpby(weight.count(), lr * wc, weight.gpu_data(), momentum_, his.mutable_gpu_data());
		cublas.axpy(weight.count(), lr, weight.gpu_diff(), his.mutable_gpu_data());
		cublas.axpy(weight.count(), (data_type)-1.f, his.gpu_data(), weight.mutable_gpu_data());
	}
}