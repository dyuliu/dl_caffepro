
#pragma once

#include <caffepro/updater/sgd_updater.h>

namespace caffepro {
	class sgd_updater_faster : public sgd_updater {
	public:
		// definations
		struct update_group {
			// do not call cpu functions after initialization!!! (act as gpu buffer only)
			boost::shared_ptr<node_blob> group_data;
			// do not call cpu functions after initialization!!! (act as gpu buffer only)
			boost::shared_ptr<device_blob> group_history;
			std::vector<int> member_indexes;
			data_type local_lr;
			data_type local_wc;
		};

	public:
		sgd_updater_faster(caffepro_context *context, caffepro_net::weight_info &weight_info, SolverParameter &param, std::vector<metric> &update_metrics);
		virtual ~sgd_updater_faster();

	protected:
		// overrides
		virtual void on_update_first_device(int param_id, data_type lr, data_type wc);
		virtual void merge_diff(bool ignore_zero_lr);
		virtual void broadcast_data(bool ignore_zero_lr);

	public:
		// override interface
		virtual void update(data_type global_lr, data_type global_wc, bool ignore_zero_lr = true);

	protected:
		void init();

	protected:
		std::vector<update_group> update_groups_;
	};
}