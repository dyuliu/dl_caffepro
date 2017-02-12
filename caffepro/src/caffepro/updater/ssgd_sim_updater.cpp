
#include <caffepro/updater/ssgd_sim_updater.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/color_print.h>

namespace caffepro {
	ssgd_sim_updater::ssgd_sim_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics)
		: sgd_updater(context, weight_info, param, update_metrics), momentum_(param.momentum()), update_interval_(param.sim_update_interval()) {

		COUT_METD << "Using synchronized sgd method with simulation" << std::endl;

		size_ = 0;
		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			size_ += weight_info_.weights[i]->get(0)->count();
		}

		param_.reset(device_blob::create_4d(
			context_,
			size_, 1, 1, 1,
			weight_info_.weights[0]->get(0)->device_id()
			));

		multinode_ = multinode::get();

		iter_ = 0;
		worker_num_ = multinode_->get_worker_size();
		worker_id_ = multinode_->get_worker_id();
		update_interval_count_ = 0;

		init();
	}

	ssgd_sim_updater::~ssgd_sim_updater() {
	}

	void _sum_content(data_type* _data, size_t _size, std::string show, int iter_, int worker_id_ = 0) {
		double sum = 0;
		for (int i = 0; i < _size; i++) {
			sum += _data[i];
		}
		COUT_WORKER(worker_id_, show) << sum << " in " << iter_ << std::endl;
	}

	void ssgd_sim_updater::load_updater_state(SolverState &state) {
		// load solver state
		CHECK_EQ(state.history_size(), history_.size()) << "Incorrect length of history blobs.";
		LOG(INFO) << "Sychronized SGDSolver: restoring history";

		for (int i = 0; i < (int)history_.size(); ++i) {
			history_[i]->load_data_from(state.mutable_history(i));
		}
		COUT_SUCC << "Finished the load solverstate of SSGD" << std::endl;
	}

	void ssgd_sim_updater::save_updater_state(SolverState &state) {
		state.clear_history();
		for (int i = 0; i < history_.size(); ++i) {
			// Add history
			BlobProto* history_blob = state.add_history();
			history_[i]->save_data_to(history_blob);
		}
	}

	void ssgd_sim_updater::init() {
		COUT_WORKER(worker_id_, "Prepare to synchronize all models") << std::endl;
		clock_t start_time = clock();
		get_model_data(param_);
		multinode_->all_sum(param_->mutable_cpu_data(), size_);
		for (int i = 0; i < size_; i++) param_->mutable_cpu_data()[i] /= worker_num_;
		set_model_data(param_);
		param_->fill_diff(0);
		COUT_WORKER(worker_id_, "Finish Model Synchronizing, Using ") << clock() - start_time << "ms" << std::endl;
	}

	void ssgd_sim_updater::get_his_data(boost::shared_ptr<device_blob> &param_) {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &his = *history_[i];
			cublas.copy(his.count(), his.gpu_data(), param_->mutable_gpu_data() + count);
			count += his.count();
		}
	}

	void ssgd_sim_updater::get_model_data(boost::shared_ptr<device_blob> &param_) {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), weight.gpu_data(), param_->mutable_gpu_data() + count);
			count += weight.count();
		}
	}

	void ssgd_sim_updater::set_model_data(boost::shared_ptr<device_blob> &param_) {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), param_->gpu_data() + count, weight.mutable_gpu_data());
			count += weight.count();
		}
	}

	void ssgd_sim_updater::get_model_diff(boost::shared_ptr<device_blob> &param_) {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), weight.gpu_diff(), param_->mutable_gpu_diff() + count);
			count += weight.count();
		}
	}

	void ssgd_sim_updater::set_model_diff(boost::shared_ptr<device_blob> &param_) {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), param_->gpu_diff() + count, weight.mutable_gpu_diff());
			count += weight.count();
		}
	}

	void ssgd_sim_updater::sync_diff() {
		// get_model_diff();
		multinode_->all_sum(param_->mutable_cpu_diff(), size_);
		for (int i = 0; i < size_; i++) param_->mutable_cpu_diff()[i] /= (worker_num_*update_interval_);
		set_model_diff(param_);
		context_->sync_all_devices();
	}

	void ssgd_sim_updater::add_model_diff(boost::shared_ptr<device_blob> &param_) {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.axpy(weight.count(), 1.0, weight.gpu_diff(), param_->mutable_gpu_diff() + count);
			count += weight.count();
		}
	}

	void ssgd_sim_updater::update(data_type global_lr, data_type global_wc, bool ignore_zero_lr) {

		if (!ignore_zero_lr || global_lr != 0) {
			// collect grad info
			merge_diff(ignore_zero_lr);

			add_model_diff(param_);
			update_interval_count_++;

			if (update_interval_count_ == update_interval_) {

				// ssgd
				sync_diff();

				if (worker_id_) global_lr = 0;
				multinode_->all_sum(&global_lr, 1);

				for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
					data_type lr = weight_info_.learning_rate[i] * global_lr;
					data_type wc = weight_info_.weight_decay[i] * global_wc;

					if (!ignore_zero_lr || lr != 0) {
						on_update_first_device(i, lr, wc);
					}
				}

				context_->sync_all_devices();
				broadcast_data(ignore_zero_lr);

				iter_++;

				// clear
				update_interval_count_ = 0;
				param_->fill_diff(0);
			};
		}
	}
}