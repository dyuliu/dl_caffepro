
#pragma once

#include <caffepro/updater/sgd_updater.h>
#include <caffepro/utils/multinode.h>

namespace caffepro {
	class record_sim_ssgd_updater : public sgd_updater {
	public:
		record_sim_ssgd_updater(
			caffepro_context *context,
			caffepro_net::weight_info &weight_info,
			SolverParameter &param,
			std::vector<metric> &update_metrics,
			std::vector<boost::shared_ptr<caffepro_layer>> &layers,
			int &cur_iter,
			int &ori_iter,
			boost::shared_ptr<data_model::data_provider> &data_provider
		);
		virtual ~record_sim_ssgd_updater();

	public:
		// interfaces
		virtual void load_updater_state(SolverState &state);
		virtual void save_updater_state(SolverState &state);
		virtual void update(data_type global_lr, data_type global_wc, bool ignore_zero_lr);

	protected:
		// overrides
		virtual void get_model_diff(boost::shared_ptr<device_blob> &param_);
		virtual void set_model_diff(boost::shared_ptr<device_blob> &param_);

		virtual void get_his_data(boost::shared_ptr<device_blob> &param_);

		virtual void get_model_data(boost::shared_ptr<device_blob> &param_);
		virtual void set_model_data(boost::shared_ptr<device_blob> &param_);

		//
		virtual void add_model_diff(boost::shared_ptr<device_blob> &param_);

		void record_sim_ssgd_updater::save_record(int type);
		// MPI
		void init();
		void sync_diff();

	protected:
		boost::shared_ptr<device_blob> param_;
		data_type momentum_;
		size_t size_;
		int &iter_;
		int &ori_iter_;

		// Multinode
		multinode* multinode_;
		int worker_num_;
		int worker_id_;
		const int update_interval_;
		int update_interval_count_;

		// Record
		int dump_invl_;
		std::vector<boost::shared_ptr<caffepro_layer>> layers_;
		boost::shared_ptr<data_model::data_provider> data_provider_;

	private:
		DISABLE_COPY_AND_ASSIGN(record_sim_ssgd_updater);
	};
}