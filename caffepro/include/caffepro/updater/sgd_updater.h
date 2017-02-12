
#pragma once

#include <caffepro/updater/updater.h>

namespace caffepro {
	class sgd_updater : public updater {

	public:
		sgd_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics);
		virtual ~sgd_updater();

	public:
		// interfaces
		virtual void load_updater_state(SolverState &state);
		virtual void save_updater_state(SolverState &state);

	protected:
		// overrides
		virtual void on_update_first_device(int param_id, data_type lr, data_type wc);

	protected:
		std::vector<boost::shared_ptr<device_blob> > history_;
		data_type momentum_;

	private:
		DISABLE_COPY_AND_ASSIGN(sgd_updater);
	};
}