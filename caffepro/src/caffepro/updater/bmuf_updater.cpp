
#include <caffepro/updater/bmuf_updater.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/color_print.h>
#include <chrono>

#include <ppl.h>

#define TIME_OUTPUT

#ifdef TIME_OUTPUT
#define TIME_OUTPUT_SPEND(str, func) {auto t_start = clock(); (func); update_metrics_.push_back({"updater-bmuf", str, double(clock()-t_start)});}
#else
#define TIME_OUTPUT_SPEND(str, func) {(func)}
#endif
namespace caffepro {
	bmuf_updater::bmuf_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &solverparam, std::vector<metric> &update_metrics)
		: sgd_updater(context, weight_info, solverparam, update_metrics), momentum_(solverparam.momentum()), update_metrics_(update_metrics) {

		COUT_METD << "Using BMUF method." << std::endl;

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

		// MPI init
		init();

		// BMUF related
		bmuf_init(solverparam);
	}

	bmuf_updater::~bmuf_updater() {
		delete[] block_global_;
		delete[] block_delta_;
		delete[] block_params_;
		delete[] block_grad_;
	}

	void bmuf_updater::load_updater_state(SolverState &state) {
		// load solver state
		CHECK_EQ(state.history_size(), history_.size()) << "Incorrect length of history blobs.";
		LOG(INFO) << "Sychronized SGDSolver: restoring history";

		for (int i = 0; i < (int)history_.size(); ++i) {
			history_[i]->load_data_from(state.mutable_history(i));
		}
		COUT_SUCC << "Finished the load solverstate of BMUF" << std::endl;
	}

	void bmuf_updater::save_updater_state(SolverState &state) {
		state.clear_history();
		for (int i = 0; i < history_.size(); ++i) {
			// Add history
			BlobProto* history_blob = state.add_history();
			history_[i]->save_data_to(history_blob);
		}
	}

	void bmuf_updater::bmuf_init(SolverParameter &solverparam) {
        // configure the BMUF parameter
		block_method_ =  solverparam.bmuf_method();
		block_momentum_ = solverparam.bmuf_momentum();
		block_lr_ = solverparam.bmuf_lr();
		block_interval_ = solverparam.bmuf_interval();

        // initialize BMUF related
		block_global_ = new data_type[size_];
		block_delta_ = new data_type[size_];
		block_params_ = new data_type[size_];
		block_grad_ = new data_type[size_];

		// initialize values
		for (int i = 0; i < size_; i++) {
			block_params_[i] = param_->cpu_data()[i];
			block_global_[i] = block_params_[i];
			block_grad_[i] = 0;
			block_delta_[i] = 0;
		}

        // output
        COUT_WORKID(worker_id_) << "BMUF method: " << block_method_ 
                  << ", lr: " << block_lr_ 
                  << ", momentum: " << block_momentum_ 
                  << ", interval: " << block_interval_ << std::endl;
		COUT_WORKID(worker_id_) << "All parameters have been initialized, Start to run" << std::endl;
    }

	void bmuf_updater::init() {
		COUT_WORKER(worker_id_, "Prepare to synchronize all models") << std::endl;
		clock_t start_time = clock();
		get_model_data();
		multinode_->all_sum(param_->mutable_cpu_data(), size_);
		for (int i = 0; i < size_; i++) param_->mutable_cpu_data()[i] /= worker_num_;
		set_model_data();
		COUT_WORKER(worker_id_, "Finish Model Synchronizing, Using ") << clock()-start_time << "ms" << std::endl;
	}

	void bmuf_updater::get_his_data() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &his = *history_[i];
			cublas.copy(his.count(), his.gpu_data(), param_->mutable_gpu_data() + count);
			count += his.count();
		}
	}

	void bmuf_updater::get_model_data() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), weight.gpu_data(), param_->mutable_gpu_data() + count);
			count += weight.count();
		}
	}

	void bmuf_updater::set_model_data() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), param_->gpu_data() + count, weight.mutable_gpu_data());
			count += weight.count();
		}
	}

	void bmuf_updater::get_model_diff() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			cublas.copy(weight.count(), weight.gpu_diff(), param_->mutable_gpu_diff() + count);
			count += weight.count();
		}
	}

	void bmuf_updater::set_model_diff() {
		cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
		for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			// cublas.axpby(weight.count(), 1 / worker_num_, param_->gpu_diff() + count, 0, weight.mutable_gpu_diff());
			cublas.copy(weight.count(), param_->gpu_diff() + count, weight.mutable_gpu_diff());
			count += weight.count();
		}
	}

	void bmuf_updater::update(data_type global_lr, data_type global_wc, bool ignore_zero_lr) {

		if (!ignore_zero_lr || global_lr != 0) {

			merge_diff(ignore_zero_lr);

			for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
				data_type lr = weight_info_.learning_rate[i] * global_lr;
				data_type wc = weight_info_.weight_decay[i] * global_wc;

				if (!ignore_zero_lr || lr != 0) {
					on_update_first_device(i, lr, wc);
				}
			}

			iter_++;

			// --------------------------- BMUF ------------------------------

			auto t_start = clock();
			clock_t t_start_1 = 0, t_start_2 = 0, t_start_3 = 0, t_start_4 = 0;
			clock_t t_end_1 = 0, t_end_2 = 0, t_end_3 = 0, t_end_4 = 0;
			
			if (iter_ % block_interval_ == 0) {

				// 1. copy GPU data to CPU memory
				t_start_1 = clock();
				cublas_wrapper<data_type> cublas(context_, weight_info_.weights[0]->get(0)->device_id());
				for (int i = 0, count = 0; i < (int)weight_info_.weights.size(); i++) {
					device_blob &weight = *weight_info_.weights[i]->get(0);
					device_blob &his = *history_[i];
					cublas.copy(weight.count(), weight.gpu_data(), param_->mutable_gpu_data() + count);
					// 2. clear the histroy momentum info
					cublas.fill_constant(weight.count(), 0.f, his.mutable_gpu_data());
					count += weight.count();
				}
				context_->sync_all_devices();
				t_end_1 = clock();
				
				// 3. all sum reduce
				t_start_2 = clock();
				multinode_->all_sum(param_->mutable_cpu_data(), size_);
				t_end_2 = clock();

				// 4. compute values
				t_start_3 = clock();
				Concurrency::parallel_for(size_t(0), size_, [&](size_t i) {
					block_grad_[i] = param_->mutable_cpu_data()[i] / worker_num_ - block_global_[i];
					block_delta_[i] = block_momentum_ * block_delta_[i] + block_lr_ * block_grad_[i];
					block_params_[i] = block_params_[i] + block_delta_[i];
					if (block_method_ == "CBM") block_global_[i] = block_params_[i];
					if (block_method_ == "NBM") block_global_[i] = block_params_[i] + block_momentum_ * block_delta_[i];
					param_->mutable_cpu_data()[i] = block_global_[i];
				});
				t_end_3 = clock();

				// 5. copy to model
				t_start_4 = clock();
				set_model_data();
				t_end_4 = clock();
			}

			update_metrics_.push_back({ "updater-bmuf", "bmuf_gpu_to_cpu_time", double(t_end_1 - t_start_1) });
			update_metrics_.push_back({ "updater-bmuf", "bmuf_mpi_time", double(t_end_2 - t_start_2) });
			update_metrics_.push_back({ "updater-bmuf", "bmuf_update_time", double(t_end_3 - t_start_3) });
			update_metrics_.push_back({ "updater-bmuf", "bmuf_cpu_to_gpu_time", double(t_end_4 - t_start_4) });
			update_metrics_.push_back({ "updater-bmuf", "bmuf_sum_time", double(clock() - t_start) });
			
			// ---------------------------- END ---------------------------------

			context_->sync_all_devices();
			broadcast_data(ignore_zero_lr);
		}
	}
}