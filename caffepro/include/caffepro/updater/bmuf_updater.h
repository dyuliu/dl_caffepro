
#pragma once

#include <caffepro/updater/sgd_updater.h>
#include <caffepro/utils/multinode.h>

namespace caffepro {
	class bmuf_updater : public sgd_updater {
	public:
		bmuf_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics);
		virtual ~bmuf_updater();

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
		virtual void get_model_diff();
		virtual void set_model_diff();

		virtual void get_his_data();

		virtual void get_model_data();
		virtual void set_model_data();

		// MPI
		void init();
		void bmuf_init(SolverParameter &solverparam);
		void sync_diff();

        // BMUF

	protected:
		boost::shared_ptr<device_blob> param_;
		data_type momentum_;
		size_t size_;
		multinode* multinode_;
		size_t iter_;
		int worker_num_;
		int worker_id_;

		std::vector<metric> &update_metrics_;

        // bmuf related
        std::string block_method_; // CBM or NBM        
        data_type block_momentum_;
        data_type block_lr_;
        int block_interval_;
        // bmuf memory
		data_type *block_global_;
		data_type *block_delta_;
		data_type *block_params_;
		data_type *block_grad_;


	private:
		DISABLE_COPY_AND_ASSIGN(bmuf_updater);
	};
}