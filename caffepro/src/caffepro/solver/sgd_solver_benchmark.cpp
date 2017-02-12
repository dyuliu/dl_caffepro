
#include <caffepro/solver/sgd_solver_benchmark.h>
#include <caffepro/proto/proto_io.h>

#include <iostream>

namespace caffepro {
	using std::vector;
	using std::string;
	using std::cout;
	using std::endl;

	sgd_solver_benchmark::sgd_solver_benchmark(caffepro_context *context, const std::string &solver_param_file,
		boost::shared_ptr<data_model::data_provider> dataprovider)
		: context_(context), iter_(0) {
		CHECK(context_);

		proto_io(param_).from_text_file(solver_param_file);
		init(dataprovider);
	}

	sgd_solver_benchmark::sgd_solver_benchmark(caffepro_context *context, SolverParameter &solver_param,
		boost::shared_ptr<data_model::data_provider> dataprovider)
		: context_(context), param_(solver_param), iter_(0) {
		CHECK(context_);

		init(dataprovider);
	}

	void sgd_solver_benchmark::init(boost::shared_ptr<data_model::data_provider> dataprovider) {
		context_->set_phase(caffepro_context::TRAIN);
		solver_device_id_ = param_.device_id();

		NetParameter train_net_param;
		proto_io(train_net_param).from_text_file(param_.train_net());
		LOG(INFO) << "Creating training net.";

		ENTER_DEVICE_CONTEXT(solver_device_id_)
			net_.reset(caffepro_net::create_from_proto(context_, train_net_param, dataprovider));
			updater_.reset(new sgd_updater(context_, net_->weights_info(), param_, update_metrics));
		EXIT_DEVICE_CONTEXT;
		LOG(INFO) << "Solver scaffolding done.";
	}

	void sgd_solver_benchmark::load_net_param_only(NetParameter &param) {
		net_->load_weights(param);
	}

	void sgd_solver_benchmark::load_net_param_only(const std::string &param_file) {
		NetParameter param;
		proto_io(param).from_binary_file(param_file);
		load_net_param_only(param);
	}

	void sgd_solver_benchmark::benchmark(bool show_details) {
		ENTER_DEVICE_CONTEXT(solver_device_id_)
			int next_test_iter = INT_MAX;
			int next_snapshot_iter = iter_ + param_.snapshot();

			data_type best_error = FLT_MAX;
			while (iter_ < param_.max_iter() && !context_->get_signal(caffepro_context::SIGNAL_STOP_ALL)) {
				int training_cycle = std::min(param_.max_iter() - iter_, param_.display());

				vector<metric> training_metrics;
				data_type training_error = train(training_cycle, training_metrics, show_details);
				iter_ += training_cycle;
				display_metrics(training_error, training_metrics, "TRAIN");
			}
		EXIT_DEVICE_CONTEXT;
	}

	data_type sgd_solver_benchmark::train(int iterations, std::vector<metric> &metrics, bool show_details) {
		metrics.clear();
		context_->set_phase(caffepro_context::TRAIN);

		data_type global_lr = 0, global_wc = param_.weight_decay();
		double total_fw_time = 0, total_bp_time = 0, total_update_time = 0;
		for (int i = 0; i < iterations; i++) {
			clock_t start_time = clock();
			if (show_details) {
				net_->forward_benchmark(false);
			}
			context_->sync_all_devices();
			int forward_time = clock() - start_time;

			start_time = clock();
			if (show_details) {
				net_->backward_benchmark(true);
			}
			context_->sync_all_devices();
			int backward_time = clock() - start_time;

			start_time = clock();
			global_lr = (data_type)1.f;
			updater_->update(global_lr, global_wc, true);
			context_->sync_all_devices();
			int update_time = clock() - start_time;

			merge_metrics(metrics, *net_);
			total_fw_time += forward_time;
			total_bp_time += backward_time;
			total_update_time += update_time;
		}

		for (int i = 0; i < (int)metrics.size(); i++) {
			metrics[i].value /= iterations;
		}

		metric fw_t = { "solver", "Forward time", total_fw_time / iterations };
		metric bp_t = { "solver", "Backward time", total_bp_time / iterations };
		metric up_t = { "solver", "Update time", total_update_time / iterations };
		metric lr = { "solver", "Learning rate", global_lr };
		metrics.push_back(fw_t);
		metrics.push_back(bp_t);
		metrics.push_back(up_t);
		metrics.push_back(lr);

		if (param_.has_train_primary_output_index()) {
			CHECK_LT(param_.train_primary_output_index(), metrics.size());
			return (data_type)metrics[param_.train_primary_output_index()].value;
		}

		return (data_type)metrics[0].value;
	}

	void sgd_solver_benchmark::merge_metrics(std::vector<metric> &metrics, caffepro_net &net) {
		auto &net_outputs = net.output_blobs();

		for (int i = 0, index = 0; i < (int)net_outputs.size(); i++) {
			if (net_outputs[i]->output_bindings().size() > 0) {
				CHECK_LE(net_outputs[i]->tags().size(), net_outputs[i]->get(0)->count());
				string src_name = net_outputs[i]->output_bindings()[0]->layer_param().name();
				const data_type *metric_data = net_outputs[i]->get(0)->cpu_data();
				for (int j = 0; j < (int)net_outputs[i]->tags().size(); j++, index++) {
					string tag = net_outputs[i]->tags()[j];

					if (index >= metrics.size()) {
						metric met = { src_name, tag, metric_data[j] };
						metrics.push_back(met);
					}
					else {
						CHECK_EQ(src_name, metrics[index].source);
						CHECK_EQ(tag, metrics[index].name);
						metrics[index].value += metric_data[j];
					}
				}
			}
		}
	}

	void sgd_solver_benchmark::display_metrics(data_type error, std::vector<metric> &metrics, const std::string prefix) {
		LOG(INFO) << "--" << prefix << ": " << "iter = " << iter_ << ", error = " << error;
		for (int i = 0; i < (int)metrics.size(); i++) {
			LOG(INFO) << "(" << metrics[i].source << ") " << metrics[i].name << ": " << metrics[i].value;
		}
		LOG(INFO) << "";
		::google::FlushLogFiles(0);
	}
}