
#pragma once

#include <caffepro/updater/sgd_updater.h>
#include <caffepro/utils/multinode.h>

namespace caffepro {
	class ssgd_sim_updater : public sgd_updater {
	public:
		ssgd_sim_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics);
		virtual ~ssgd_sim_updater();

	public:
		// interfaces
		virtual void load_updater_state(SolverState &state);
		virtual void save_updater_state(SolverState &state);
		virtual void update(data_type global_lr, data_type global_wc, bool ignore_zero_lr);

		// data interface
		virtual int get_worker_id() { return worker_id_; }
		virtual std::string get_worker_id_str() { return std::to_string(worker_id_); }

	protected:
		// overrides
		virtual void get_model_diff(boost::shared_ptr<device_blob> &param_);
		virtual void set_model_diff(boost::shared_ptr<device_blob> &param_);

		virtual void get_his_data(boost::shared_ptr<device_blob> &param_);

		virtual void get_model_data(boost::shared_ptr<device_blob> &param_);
		virtual void set_model_data(boost::shared_ptr<device_blob> &param_);

		//
		virtual void add_model_diff(boost::shared_ptr<device_blob> &param_);

		// MPI
		void init();
		void sync_diff();

	protected:
		boost::shared_ptr<device_blob> param_;
		data_type momentum_;
		size_t size_;
		multinode* multinode_;
		size_t iter_;
		int worker_num_;
		int worker_id_;
		const int update_interval_;
		int update_interval_count_;

	private:
		DISABLE_COPY_AND_ASSIGN(ssgd_sim_updater);
	};
}