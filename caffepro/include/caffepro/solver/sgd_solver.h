
#pragma once 

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>
#include <caffepro/proto/caffe.pb.h>
#include <boost/shared_array.hpp>
#include <caffepro/data/data_accessors/sequential_accessor_ssgd.h>

#include <caffepro/utils/analyzer.h>

namespace caffepro {
	namespace data_model {
		class data_provider;
	}

	class caffepro_net;
	class updater;

	class sgd_solver : public caffepro_object {

	protected:
		using metric = caffepro::caffepro_object::output_metric;

	public:
		sgd_solver(
			caffepro_context *context, 
			const std::string &solver_param_file,
			boost::shared_ptr<data_model::data_provider> dataprovider_train = nullptr,
			boost::shared_ptr<data_model::data_provider> dataprovider_test = nullptr,
			const std::string &default_updater = "SGD"
			);

		sgd_solver(
			caffepro_context *context, 
			SolverParameter &solver_param,
			boost::shared_ptr<data_model::data_provider> dataprovider_train = nullptr,
			boost::shared_ptr<data_model::data_provider> dataprovider_test = nullptr,
			const std::string &default_updater = "SGD"
			);

		~sgd_solver();

	public:
		// interfaces
		void load(SolverState &param, bool load_solver_state = true);
		void load(const std::string &param_file, bool load_solver_state = true);
		void load_net_param_only(NetParameter &param);
		void load_net_param_only(const std::string &param_file);
		void save(SolverState &solver_param, NetParameter &net_param);
		void save(const std::string &solver_param_file, const std::string &net_param_file);
		void snapshot(const std::string &prefix = "");
		void run();

		int get_iter() const { return iter_; }
		data_type get_learning_rate() { return get_learning_rate(iter_); }

	public:
		// fetch functions
		caffepro_context *context() const { return context_; }
		boost::shared_ptr<caffepro_net> net() { return net_; }
		boost::shared_ptr<caffepro_net> test_net() { return test_net_; }
		boost::shared_ptr<updater> get_updater() { return updater_; }
		void set_updater(boost::shared_ptr<updater> updater) { updater_ = updater; }

		// data model
		void init_data_provider();
		boost::shared_ptr<data_model::data_accessor> original_accessor_;
		boost::shared_ptr<data_model::sequential_accessor_ssgd> seq_ssgd_accessor_;

	protected:
		virtual void init(
			boost::shared_ptr<data_model::data_provider> dataprovider_train,
			boost::shared_ptr<data_model::data_provider> dataprovider_test,
			const std::string &default_updater
			);
		virtual data_type train(int iterations, int running_iterations, std::vector<metric> &metrics);
		virtual data_type test(std::vector<metric> &metrics);
		virtual void merge_metrics(std::vector<metric> &metrics, caffepro_net &net);
		virtual void display_metrics(data_type error, std::vector<metric> &metrics, const std::string prefix);
		virtual data_type get_learning_rate(int cur_iter);

	protected:
		caffepro_context *context_;
		SolverParameter param_;
		int iter_;
		int ori_iter_;
		int cur_iter_;
		int solver_device_id_;
		boost::shared_ptr<caffepro_net> net_, test_net_;
		boost::shared_ptr<updater> updater_;
		int worker_id_;

		// record more information
		std::vector<metric> update_metrics;
		std::vector<metric> update_mean_metrics;

		//
		std::string cur_sgd_method;

		// running info
		analyzer::RecordInfo recorder_;

	private:
		DISABLE_COPY_AND_ASSIGN(sgd_solver);
	};
}