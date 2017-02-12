
#pragma once 

#include <caffepro/solver/sgd_solver.h>
#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>
#include <caffepro/proto/caffe.pb.h>
#include <boost/shared_array.hpp>

namespace caffepro {

	class sgd_solver_ohem : public sgd_solver {
	public:
		sgd_solver_ohem(
			caffepro_context *context,
			const std::string &solver_param_file,
			boost::shared_ptr<data_model::data_provider> dataprovider_train = nullptr,
			boost::shared_ptr<data_model::data_provider> dataprovider_test = nullptr,
			const std::string &default_updater = "SGD"
			);

		sgd_solver_ohem(
			caffepro_context *context,
			SolverParameter &solver_param,
			boost::shared_ptr<data_model::data_provider> dataprovider_train = nullptr,
			boost::shared_ptr<data_model::data_provider> dataprovider_test = nullptr,
			const std::string &default_updater = "SGD"
			);
	
	protected:
		virtual data_type train(int iterations, int running_iterations, std::vector<metric> &metrics);
		
	};
}