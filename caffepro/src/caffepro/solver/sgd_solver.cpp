
#include <caffepro/solver/sgd_solver.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/string_uitls.h>
#include <caffepro/updater/updaters.h>
#include <caffepro/utils/multinode.h>
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/layers/softmax_loss_layer.h>
#include <caffepro/utils/analyzer.h>

#include <iostream>

namespace caffepro {
	using std::vector;
	using std::string;
	using std::cout;
	using std::endl;

	sgd_solver::sgd_solver(caffepro_context *context, const std::string &solver_param_file, 
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: context_(context), iter_(0) {
		CHECK(context_);

		worker_id_ = multinode::get()->get_worker_id();

		//auto recorder_list = filesystem::get_files("./", std::string("*_" + std::to_string(worker_id_) + ".recorder").c_str(), false);
		//if (recorder_list.size()) {
		//	COUT_CHEK << "Load recorder from file: " << recorder_list[0] << std::endl;
		//	recorder_.load_from_file(recorder_list[0]);
		//}

		proto_io(param_).from_text_file(solver_param_file);

		cur_sgd_method = "";
		cur_iter_ = 0;
		init(dataprovider_train, dataprovider_test, default_updater);

	}

	sgd_solver::~sgd_solver() {
	}

	sgd_solver::sgd_solver(caffepro_context *context, SolverParameter &solver_param,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: context_(context), param_(solver_param), iter_(0) {
		CHECK(context_);

		init(dataprovider_train, dataprovider_test, default_updater);
	}

	void sgd_solver::init(
		boost::shared_ptr<data_model::data_provider> dataprovider_train, 
		boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater) {
		context_->set_phase(caffepro_context::TRAIN);
		solver_device_id_ = param_.device_id();

		NetParameter train_net_param;
		proto_io(train_net_param).from_text_file(param_.train_net());

		//recorder_.set_name(train_net_param.name());
		LOG(INFO) << "Creating training net.";

		// DeepTracker-5: initialize Analyzer_tools
		analyzer_tools_instance_.reset(new analyzer_tools::Analyzer("finaltlib2", "ig-1x-lr2", "localhost:27017"));

		ENTER_DEVICE_CONTEXT(solver_device_id_)
			net_.reset(caffepro_net::create_from_proto(context_, train_net_param, dataprovider_train));
			//
			cur_sgd_method = default_updater;
			// create updater
			if (string_equal_ignorecase(default_updater, "SGD")) {
				updater_.reset(new sgd_updater(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (string_equal_ignorecase(default_updater, "SGD_faster")) {
				updater_.reset(new sgd_updater_faster(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (string_equal_ignorecase(default_updater, "SGD_legacy")) {
				updater_.reset(new sgd_updater_legacy(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (string_equal_ignorecase(default_updater, "nesterov")) {
				updater_.reset(new nesterov_updater(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (string_equal_ignorecase(default_updater, "Record_SIM_SSGD")) {
				if (param_.data_split()) init_data_provider();
				updater_.reset(new record_sim_ssgd_updater(context_, net_->weights_info(), param_, update_metrics, 
					net_->layers(), analyzer_tools_instance_, cur_iter_, ori_iter_, net_->data_provider()));
			}
			else if (string_equal_ignorecase(default_updater, "SIM_SSGD")) {
				if (param_.data_split()) init_data_provider();
				updater_.reset(new ssgd_sim_updater(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (string_equal_ignorecase(default_updater, "SSGD")) {
				if (param_.data_split()) init_data_provider();
				updater_.reset(new ssgd_updater(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (string_equal_ignorecase(default_updater, "BMUF")) {
				if (param_.data_split()) init_data_provider();
				updater_.reset(new bmuf_updater(context_, net_->weights_info(), param_, update_metrics));
			}
			else if (default_updater.empty()) {
				LOG(ERROR) << "You do not specify an updater. Use the default one (SGD without momentum)";
				updater_.reset(new updater(context_, net_->weights_info()));
			}
			else {
				LOG(FATAL) << "Unknown updater: " << default_updater;
			}

			if (param_.has_test_net()) {
				context_->set_phase(caffepro_context::TEST);

				LOG(INFO) << "Creating testing net.";
				NetParameter test_net_param;
				proto_io(test_net_param).from_text_file(param_.test_net());

				test_net_.reset(caffepro_net::create_from_proto(context_, test_net_param, dataprovider_test));
				test_net_->share_weights_from(*net_);

				CHECK_GT(param_.test_iter(), 0);
				CHECK_GT(param_.test_interval(), 0);

				context_->set_phase(caffepro_context::TRAIN);
			}
		EXIT_DEVICE_CONTEXT;

		context_->set_shared_object(net_->get_name(), SHAREDOBJNAME_SGDSOLVER, this);
		
		LOG(INFO) << "Solver scaffolding done.";
	}

	void sgd_solver::init_data_provider() {
		CHECK(net_->data_provider()) << "To use ssgd, data provider must be provided";

		COUT_METD << "Using data split method for sychronized batch" << std::endl;

		seq_ssgd_accessor_.reset(new data_model::sequential_accessor_ssgd(*net_->data_provider()));
		seq_ssgd_accessor_->init();

		original_accessor_ = net_->data_provider()->get_data_accessor();
		net_->data_provider()->set_data_accessor(seq_ssgd_accessor_);
		net_->data_provider()->begin_prefetch();
	}

	void sgd_solver::load(SolverState &param, bool load_solver_state) {
		iter_ = param.iter();

		if (param.has_learned_net()) {
			load_net_param_only(param.learned_net());
		}

		if (load_solver_state) {
			updater_->load_updater_state(param);
		}
	}

	void sgd_solver::load(const std::string &param_file, bool load_solver_state) {
		SolverState state;
		proto_io(state).from_binary_file(param_file);
		load(state, load_solver_state);
		// continue to load recorder
	}

	void sgd_solver::load_net_param_only(NetParameter &param) {
		net_->load_weights(param);
	}

	void sgd_solver::load_net_param_only(const std::string &param_file) {
		NetParameter param;
		proto_io(param).from_binary_file(param_file);
		load_net_param_only(param);
	}

	void sgd_solver::save(SolverState &solver_param, NetParameter &net_param) {
		net_->save_proto(net_param);
		solver_param.set_iter(iter_);
		updater_->save_updater_state(solver_param);
	}

	void sgd_solver::save(const std::string &solver_param_file, const std::string &net_param_file) {
		SolverState solver_param;
		NetParameter net_param;
		save(solver_param, net_param);
		solver_param.set_learned_net(net_param_file);

		proto_io(net_param).to_binary_file(net_param_file);
		proto_io(solver_param).to_binary_file(solver_param_file);
	}

	void sgd_solver::snapshot(const std::string &prefix) {
		string filename = param_.snapshot_prefix();
		if (!prefix.empty()) {
			filename += prefix;
		}
		else {
			filename += "_iter_" + std::to_string(iter_) + "_id_" + std::to_string(worker_id_);
		}
		LOG(INFO) << "Snapshotting to " << filename;

		save(filename + ".solverstate", filename + ".model");
	}

	// DeepTracker-6: training process running function
	void sgd_solver::run() {
		ENTER_DEVICE_CONTEXT(solver_device_id_)
			int next_test_iter = INT_MAX;
			int next_snapshot_iter = iter_ + param_.snapshot();
			if (test_net_) {
				next_test_iter = iter_ + param_.test_interval();
			}
			
			
			data_type best_error = FLT_MAX;
			int running_iterations = 0;
			ori_iter_ = iter_;
			net_->data_provider()->img_info()->Clear();
			while (iter_ < param_.max_iter() && !context_->get_signal(caffepro_context::SIGNAL_STOP_ALL)) {
				int training_cycle = std::min(param_.max_iter() - iter_, param_.display());
				vector<metric> training_metrics;
				data_type training_error = train(training_cycle, running_iterations, training_metrics);  // train function
				iter_ += training_cycle; // or = cur_iter_
				running_iterations += training_cycle;
				display_metrics(training_error, training_metrics, "TRAIN");

				if (iter_ >= next_test_iter) {
					vector<metric> test_metrics;
					net_->release_blobs();
					test_net_->data_provider()->test_img_info()->Clear();
					data_type test_error = test(test_metrics);	// test function
					test_net_->release_blobs();
					display_metrics(test_error, test_metrics, "TEST");

					COUT_WORKID(worker_id_) << "Iteration " << iter_ << ":" 
						 << " training error = " << training_error
						 << " test error = " << test_error << endl;

					if (test_error < best_error) {
						best_error = test_error;
						snapshot("_best_id_" + std::to_string(worker_id_));
					}

					next_test_iter = iter_ + param_.test_interval();
				}

				if (iter_ >= next_snapshot_iter) {
					snapshot();
					next_snapshot_iter = iter_ + param_.snapshot();
				}
			}
			
		EXIT_DEVICE_CONTEXT;
	}

	data_type sgd_solver::train(int iterations, int running_iterations, std::vector<metric> &metrics) {
		metrics.clear();
		update_mean_metrics.clear();
		context_->set_phase(caffepro_context::TRAIN);

		bool debug_mode = (context_->get_global_cfg(GLOBALCFGNAME_DEBUG_MODE) == "TRUE");

		data_type global_lr = 0, global_wc = param_.weight_decay();
		double total_fw_time = 0, total_bp_time = 0, total_update_time = 0;
		double total_local_merge_time = 0, total_communicate_time = 0, total_mpi_time = 0;
		for (int i = 0; i < iterations; i++) {
			clock_t start_time = clock();
			if (debug_mode) {
				net_->forward_debug(false);
			}
			else {
				net_->forward(false);
			}

			context_->sync_all_devices();
			int forward_time = clock() - start_time;

			// dongyu's checking
			/*string layer_name_ = "loss";
			auto layer = net_->get_layer(layer_name_);
			CHECK_GT(layer->inputs().size(), 0);
			node_blob *source = layer->inputs()[0].get();
			if (layer->layer_param().type() == "softmax_loss") {
				source = dynamic_cast<softmax_loss_layer &>(*layer.get()).avg_prob().get();
			}
			std::cout << source->sum_num() << std::endl;
			std::cout << net_->data_provider()->img_info()->images_size() << std::endl;*/

			start_time = clock();
			bool lazy_update = ((running_iterations + i) % param_.update_interval() != 0); // not the FIRST batch for an update interval
			if (debug_mode) {
				net_->backward_debug(true, lazy_update);
			}
			else {
				net_->backward(true, lazy_update);
			}
			context_->sync_all_devices();
			int backward_time = clock() - start_time;

			start_time = clock();
			update_metrics.clear();
			cur_iter_ = iter_ + i + 1;

			// net_->data_provider()->img_info()
			if ((running_iterations + i + 1) % param_.update_interval() == 0) { // LAST batch for an update interval
				global_lr = get_learning_rate(cur_iter_);
				updater_->update(global_lr, global_wc, true);  // DeepTracker-7: update parameters and save to mongodb
				context_->sync_all_devices();
			}
			int update_time = clock() - start_time;

			merge_metrics(metrics, *net_);
			total_fw_time += forward_time;
			total_bp_time += backward_time;
			total_update_time += update_time;

			if (i == 0) {
				for (auto item : update_metrics) update_mean_metrics.push_back(item);
			}
			else {
				for (int m = 0; m < update_mean_metrics.size(); m++) {
					update_mean_metrics[m].value += update_metrics[m].value;
				}
			}
		}

		for (int i = 0; i < (int)metrics.size(); i++) {
			metrics[i].value /= iterations;
		}

		double total_time = total_fw_time + total_bp_time + total_update_time;
		metric fw_t = { "solver", "Forward time", total_fw_time / iterations };
		metric bp_t = { "solver", "Backward time", total_bp_time / iterations };
		metric up_t = { "solver", "Update time", total_update_time / iterations };
		metrics.push_back(fw_t);
		metrics.push_back(bp_t);
		metrics.push_back(up_t);

		for (auto item : update_mean_metrics) {
			item.value /= iterations;
			metrics.push_back(item);
		}

		metric sum_t = { "solver", "Sum time", total_time / iterations };
		metric lr = { "solver", "Learning rate", global_lr };
		metrics.push_back(sum_t);
		metrics.push_back(lr);


		if (param_.has_train_primary_output_index()) {
			CHECK_LT(param_.train_primary_output_index(), metrics.size());
			return (data_type)metrics[param_.train_primary_output_index()].value;
		}

		// DeepTracker-8: save training related info to mongodb
		analyzer_tools_instance_->deal_rec_info(cur_iter_, analyzer_tools::Analyzer::RECORD_TYPE::TRAIN_ERROR, (float)metrics[0].value);
		analyzer_tools_instance_->deal_rec_info(cur_iter_, analyzer_tools::Analyzer::RECORD_TYPE::TRAIN_LOSS, (float)metrics[1].value);
		analyzer_tools_instance_->deal_rec_info(cur_iter_, analyzer_tools::Analyzer::RECORD_TYPE::FORWARD_TIME, (float)fw_t.value);
		analyzer_tools_instance_->deal_rec_info(cur_iter_, analyzer_tools::Analyzer::RECORD_TYPE::BACKWARD_TIME, (float)bp_t.value);
		analyzer_tools_instance_->deal_rec_info(cur_iter_, analyzer_tools::Analyzer::RECORD_TYPE::UPDATE_TIME, (float)up_t.value);
		analyzer_tools_instance_->deal_rec_info(cur_iter_, analyzer_tools::Analyzer::RECORD_TYPE::LEARNING_RATE, (float)lr.value);

		//recorder_.add_record(cur_iter_, analyzer::RecordInfo::RECORD_TYPE::TRAIN_ERROR, (float)metrics[0].value);
		//recorder_.add_record(cur_iter_, analyzer::RecordInfo::RECORD_TYPE::TRAIN_LOSS, (float)metrics[1].value);
		//recorder_.add_record(cur_iter_, analyzer::RecordInfo::RECORD_TYPE::FORWARD_TIME, (float)fw_t.value);
		//recorder_.add_record(cur_iter_, analyzer::RecordInfo::RECORD_TYPE::BACKWARD_TIME, (float)bp_t.value);
		//recorder_.add_record(cur_iter_, analyzer::RecordInfo::RECORD_TYPE::UPDATE_TIME, (float)up_t.value);
		//recorder_.add_record(cur_iter_, analyzer::RecordInfo::RECORD_TYPE::LEARNING_RATE, (float)lr.value);

		//for (auto item : update_mean_metrics) {
		//	auto val = item.value/iterations;
		//	auto type = item.name;
		//	recorder_.add_record(cur_iter_, type, val);
		//}
		//
		//recorder_.save_to_file("running_info_" + std::to_string(worker_id_));

		return (data_type)metrics[0].value;
	}

	data_type sgd_solver::test(std::vector<metric> &metrics) {
		CHECK(test_net_);
		metrics.clear();
		context_->set_phase(caffepro_context::TEST);

		bool debug_mode = (context_->get_global_cfg(GLOBALCFGNAME_DEBUG_MODE) == "TRUE");

		// DeepTracker-9: convert img data to Image type defined by protobuf
		auto &img_info = test_net_->data_provider()->test_img_info();

		int img_count = 0;
		for (int i = 0; i < param_.test_iter(); i++) {

			if (debug_mode) {
				test_net_->forward_debug(true);
			}
			else {
				test_net_->forward(true);
			}
			context_->sync_all_devices();
			
			string layer_name_ = "loss";
			auto layer = test_net_->get_layer(layer_name_);
			CHECK_GT(layer->inputs().size(), 0);
			node_blob *source = layer->inputs()[0].get();
			if (layer->layer_param().type() == "softmax_loss") {
				source = dynamic_cast<softmax_loss_layer &>(*layer.get()).prob().get();
			}
			int feature_dim = source->get(0)->inner_count();

			// DeepTracker-9: convert img data to Image type defined by protobuf
			for (int k = 0; k < source->sum_num(); k++) {
				const data_type *data = source->get_cpu_data_across_dev(k);
				data_type max_v = -1000000;
				int max_index = -1;
				for (int j = 0; j < feature_dim; j++) { 
					//printf("%.2f ", data[j]);
					if (data[j] > max_v) {
						max_v = data[j];
						max_index = j;
					}
					img_info->mutable_images(img_count)->add_prob(data[j]); // DeepTracker-9: set up class score vector, the one with maximum value is the final anwser
				}
				img_info->mutable_images(img_count)->set_answer(max_index); // DeepTracker-9: set up answer value
				img_count++;
			}

			merge_metrics(metrics, *test_net_);
		}
		// DeepTracker-9: save img data to mongodb
		img_info->set_iteration(iter_); // set current iteration
		analyzer::DumpInfo imgInfos;
		imgInfos.testRecord(*img_info, analyzer_tools_instance_);
		img_info->Clear();

		for (int i = 0; i < (int)metrics.size(); i++) {
			metrics[i].value /= param_.test_iter();
		}

		if (param_.has_test_primary_output_index()) {
			CHECK_LT(param_.test_primary_output_index(), metrics.size());
			return (data_type)metrics[param_.test_primary_output_index()].value;
		}

		// DeepTracker-10: save test related info to mongodb
		analyzer_tools_instance_->deal_rec_info(iter_, analyzer_tools::Analyzer::RECORD_TYPE::TEST_ERROR, (float)metrics[0].value);
		analyzer_tools_instance_->deal_rec_info(iter_, analyzer_tools::Analyzer::RECORD_TYPE::TEST_LOSS, (float)metrics[1].value);

		//recorder_.add_record(iter_, analyzer::RecordInfo::RECORD_TYPE::TEST_ERROR, (float)metrics[0].value);
		//recorder_.add_record(iter_, analyzer::RecordInfo::RECORD_TYPE::TEST_LOSS, (float)metrics[1].value);
		//recorder_.save_to_file("running_info_" + std::to_string(worker_id_));

		return (data_type)metrics[0].value;
	}

	void sgd_solver::merge_metrics(std::vector<metric> &metrics, caffepro_net &net) {

		auto &net_outputs = net.output_blobs();
		for (int i = 0, index = 0; i < (int)net_outputs.size(); i++) {
			string src_name = "global";
			if (net_outputs[i]->output_bindings().size() > 0) {
				CHECK_LE(net_outputs[i]->tags().size(), net_outputs[i]->get(0)->count());
				src_name = net_outputs[i]->output_bindings()[0]->layer_param().name();
			}
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

	void sgd_solver::display_metrics(data_type error, std::vector<metric> &metrics, const std::string prefix) {
		LOG(INFO) << "--" << prefix << ": " << "iter = " << iter_ << ", error = " << error;
		for (int i = 0; i < (int)metrics.size(); i++) {
			LOG(INFO) << "(" << metrics[i].source << ") " << metrics[i].name << ": " << metrics[i].value;
		}
		LOG(INFO) << "";
		::google::FlushLogFiles(0);
	}

	data_type sgd_solver::get_learning_rate(int cur_iter) {
		data_type rate;
		const string& lr_policy = param_.lr_policy();

		if (lr_policy == "fixed") {
			rate = param_.base_lr();
		}
		else if (lr_policy == "step") {
			int current_step = cur_iter / param_.stepsize();
			rate = param_.base_lr() * pow(param_.gamma(), current_step);
		}
		else if (lr_policy == "exp") {
			rate = param_.base_lr() * pow(param_.gamma(), cur_iter);
		}
		else if (lr_policy == "inv") {
			rate = param_.base_lr() * pow(data_type(1) + param_.gamma() * cur_iter, 
				-param_.power());
		}
		else if (lr_policy == "vstep"){
			CHECK_EQ(param_.vstep_size_size(), param_.vstep_lr_size());
			int step_iter = cur_iter;
			rate = param_.vstep_lr(param_.vstep_lr_size() - 1);
			for (int i = 0; i < param_.vstep_lr_size(); i++) {
				step_iter -= param_.vstep_size(i);
				if (step_iter <= 0) {
					rate = param_.vstep_lr(i);
					break;
				}
			}
		}
		else if (lr_policy == "poly") {
			rate = this->param_.base_lr() * pow(data_type(1.) -
				(data_type(cur_iter) / data_type(param_.max_iter())),
				param_.power());
		}
		else {
			LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
		}

		return rate;
	}
}