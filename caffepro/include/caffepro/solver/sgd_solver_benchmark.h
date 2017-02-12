
#pragma once 

#include <caffepro/updater/sgd_updater.h>
#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {

	class sgd_solver_benchmark {
	
	protected:
		using metric = caffepro::caffepro_object::output_metric;

	public:
		sgd_solver_benchmark(
			caffepro_context *context,
			const std::string &solver_param_file,
			boost::shared_ptr<data_model::data_provider> dataprovider = nullptr
			);

		sgd_solver_benchmark(
			caffepro_context *context,
			SolverParameter &solver_param,
			boost::shared_ptr<data_model::data_provider> dataprovider = nullptr
			);

	public:
		// interfaces
		void load_net_param_only(NetParameter &param);
		void load_net_param_only(const std::string &param_file);
		void benchmark(bool show_details);

	public:
		// fetch functions
		caffepro_context *context() const { return context_; }
		boost::shared_ptr<caffepro_net> net() { return net_; }
		boost::shared_ptr<sgd_updater> updater() { return updater_; }

	protected:
		void init(boost::shared_ptr<data_model::data_provider> dataprovider);
		data_type train(int iterations, std::vector<metric> &metrics, bool show_details);
		void merge_metrics(std::vector<metric> &metrics, caffepro_net &net);
		void display_metrics(data_type error, std::vector<metric> &metrics, const std::string prefix);

	protected:
		caffepro_context *context_;
		SolverParameter param_;
		int iter_;
		int solver_device_id_;
		boost::shared_ptr<caffepro_net> net_;
		boost::shared_ptr<sgd_updater> updater_;

		std::vector<metric> update_metrics;

	private:
		DISABLE_COPY_AND_ASSIGN(sgd_solver_benchmark);
	};
}