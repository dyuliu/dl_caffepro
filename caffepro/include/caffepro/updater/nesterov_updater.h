
#pragma once

#include <caffepro/updater/sgd_updater_faster.h>

namespace caffepro {
	class nesterov_updater : public sgd_updater_faster {
	public:
		nesterov_updater(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics);
		virtual ~nesterov_updater();

	protected:
		// overrides
		virtual void on_update_first_device(int param_id, data_type lr, data_type wc);
	};
}