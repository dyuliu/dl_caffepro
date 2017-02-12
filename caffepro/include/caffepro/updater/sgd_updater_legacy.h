
#pragma once

#include <caffepro/updater/updater.h>

namespace caffepro {
	class SolverState;

	class sgd_updater_legacy : public updater {

	public:
		sgd_updater_legacy(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics);
		virtual ~sgd_updater_legacy();

	public:
		// interfaces
		void load_updater_state(SolverState &state);
		void save_updater_state(SolverState &state);

		// override
		virtual void update(data_type global_lr, data_type global_wc, bool ignore_zero_lr = true);
	
	protected:
		std::vector<boost::shared_ptr<device_blob> > history_;
		data_type momentum_;

	private:
		DISABLE_COPY_AND_ASSIGN(sgd_updater_legacy);
	};
}