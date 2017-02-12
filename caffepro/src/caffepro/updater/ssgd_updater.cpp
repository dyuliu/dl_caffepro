
#include <caffepro/updater/ssgd_updater.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/color_print.h>
#include <chrono>

#define TIME_OUTPUT

#ifdef TIME_OUTPUT
	#define TIME_OUTPUT_SPEND(str, func) {auto t_start = clock(); (func); update_metrics_.push_back({"updater-ssgd", str, double(clock()-t_start)});}
#else
	#define TIME_OUTPUT_SPEND(str, func) {(func)}
#endif

namespace caffepro {
	ssgd_updater::ssgd_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics)
		: sgd_updater(context, weight_info, param, update_metrics), momentum_(param.momentum()), update_metrics_(update_metrics) {

		COUT_METD << "Using synchronized sgd method." << std::endl;

#ifdef TIME_OUTPUT
		COUT_CHEK << "Using MPI Test Mode to count the elapse time." << std::endl;
#endif

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

		d_worker_num_ = 1.0 / worker_num_;

		init();
	}

	ssgd_updater::~ssgd_updater() {
	}

	void _sum(data_type* _data, size_t _size, std::string show, int iter_, int worker_id_ = 0) {
		double sum = 0;
		for (int i = 0; i < _size; i++) {
			sum += _data[i];
		}
		COUT_WORKER(worker_id_, show) << sum << " in " << iter_ << std::endl;
	}

	void ssgd_updater::load_updater_state(SolverState &state) {
		// load solver state
		CHECK_EQ(state.history_size(), history_.size()) << "Incorrect length of history blobs.";
		LOG(INFO) << "Sychronized SGDSolver: restoring history";

		for (int i = 0; i < (int)history_.size(); ++i) {
			history_[i]->load_data_from(state.mutable_history(i));
		}
		COUT_SUCC << "Finished the load solverstate of SSGD" << std::endl;
	}

	void ssgd_updater::save_updater_state(SolverState &state) {
		state.clear_history();
		for (int i = 0; i < history_.size(); ++i) {
			// Add history
			BlobProto* history_blob = state.add_history();
			history_[i]->save_data_to(history_blob);
		}
	}

	void ssgd_updater::init() {
		COUT_WORKER(worker_id_, "Prepare to synchronize all models") << std::endl;
		clock_t start_time = clock();
		get_model_data();
		multinode_->all_sum(param_->mutable_cpu_data(), size_);
		for (int i = 0; i < size_; i++) param_->mutable_cpu_data()[i] /= worker_num_;
		set_model_data();
		COUT_WORKER(worker_id_, "Finish Model Synchronizing, Using ") << clock()-start_time << "ms" << std::endl;
	}

	void ssgd_updater::get_his_data() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &his = *history_[i];
			cublas.copy(his.count(), his.gpu_data(), param_->mutable_gpu_data() + count);
			count += his.count();
		}
	}

	void ssgd_updater::get_model_data() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), weight.gpu_data(), param_->mutable_gpu_data() + count);
			count += weight.count();
		}
	}

	void ssgd_updater::set_model_data() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), param_->gpu_data() + count, weight.mutable_gpu_data());
			count += weight.count();
		}
	}

	void ssgd_updater::get_model_diff() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), weight.gpu_diff(), param_->mutable_gpu_diff() + count);
			count += weight.count();
		}
		param_->mutable_cpu_diff();
	}

	void ssgd_updater::set_model_diff() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			// cublas.copy(weight.count(), param_->gpu_diff() + count, weight.mutable_gpu_diff());
			cublas.axpby(weight.count(), d_worker_num_, param_->gpu_diff() + count, 0, weight.mutable_gpu_diff());
			// cublas.axpy(weight.count(), d_worker_num_, param_->gpu_diff() + count, weight.mutable_gpu_diff());
			count += weight.count();
		}
	}

	void ssgd_updater::sync_diff() {
		TIME_OUTPUT_SPEND("update_mpi_gpu_to_cpu_time", get_model_diff());
		TIME_OUTPUT_SPEND("update_mpi_time", multinode_->all_sum(param_->mutable_cpu_diff(), size_));
		TIME_OUTPUT_SPEND("update_mpi_cpu_to_gpu_time", set_model_diff());
	}


	void ssgd_updater::update(data_type global_lr, data_type global_wc, bool ignore_zero_lr) {

		if (!ignore_zero_lr || global_lr != 0) {

			TIME_OUTPUT_SPEND("update_local_merge_time", merge_diff(ignore_zero_lr));
			TIME_OUTPUT_SPEND("update_mpi_sum_time", sync_diff());

			TIME_OUTPUT_SPEND("update_weight", [&]() {
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
			}());
		}
	}
}