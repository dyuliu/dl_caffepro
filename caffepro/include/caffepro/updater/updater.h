
#pragma once

#include <caffepro/object_model/caffepro_net.h>

namespace caffepro {
	class SolverState;
	class caffepro_net;

	class updater {

	protected:
		using metric = caffepro::caffepro_object::output_metric;

	public:
		updater(caffepro_context *context, caffepro_net::weight_info &weight_info);
		virtual ~updater();

	public:
		virtual void update(data_type global_lr, data_type global_wc, bool ignore_zero_lr = true);

	public:
		// overrides
		virtual void load_updater_state(SolverState &state) {}
		virtual void save_updater_state(SolverState &state) {}

	protected:
		// overrides
		virtual void on_update_first_device(int param_id, data_type lr, data_type wc);
		virtual void merge_diff(bool ignore_zero_lr);
		virtual void broadcast_data(bool ignore_zero_lr);

	protected:
		caffepro_context *context_;
		caffepro_net::weight_info &weight_info_;

	private:
		DISABLE_COPY_AND_ASSIGN(updater);
	};
}