
#pragma once

#include <caffepro/solver/sgd_solver.h>

namespace caffepro {

	namespace data_model{
		class sequential_rank_accessor;
		class data_accessor;
	}

	class softmax_loss_layer;

	class sgd_solver_rank_ohem : public sgd_solver {
	public:
		sgd_solver_rank_ohem(
			caffepro_context *context,
			const std::string &solver_param_file,
			boost::shared_ptr<data_model::data_provider> dataprovider_train = nullptr,
			boost::shared_ptr<data_model::data_provider> dataprovider_test = nullptr,
			const std::string &default_updater = "SGD"
			);

		sgd_solver_rank_ohem(
			caffepro_context *context,
			SolverParameter &solver_param,
			boost::shared_ptr<data_model::data_provider> dataprovider_train = nullptr,
			boost::shared_ptr<data_model::data_provider> dataprovider_test = nullptr,
			const std::string &default_updater = "SGD"
			);

	protected:
		void init_rank_ohem();
		void update_rank();

	protected:
		// overrides
		virtual data_type train(int iterations, int running_iterations, std::vector<metric> &metrics);
		virtual data_type test(std::vector<metric> &metrics);

	protected:
		// accessors
		boost::shared_ptr<data_model::data_accessor> original_accessor_;
		boost::shared_ptr<data_model::sequential_rank_accessor> seq_rank_accessor_;
	};
}