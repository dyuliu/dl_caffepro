
#include <caffepro/solver/sgd_solver_rank_ohem.h>
#include <caffepro/data/data_accessors/sequential_rank_accessor.h>
#include <caffepro/utils/string_uitls.h>
#include <caffepro/updater/updaters.h>
#include <caffepro/layers/softmax_loss_layer.h>
#include <caffepro/context/common_names.h>

#include <iostream>

namespace caffepro {
	using std::vector;
	using std::string;

	sgd_solver_rank_ohem::sgd_solver_rank_ohem(caffepro_context *context, const std::string &solver_param_file,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: sgd_solver(context, solver_param_file, dataprovider_train, dataprovider_test, default_updater) {
	
		init_rank_ohem();
	}

	sgd_solver_rank_ohem::sgd_solver_rank_ohem(caffepro_context *context, SolverParameter &solver_param,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: sgd_solver(context, solver_param, dataprovider_train, dataprovider_test, default_updater) {
		
		init_rank_ohem();
	}
	
	void sgd_solver_rank_ohem::init_rank_ohem() {
		CHECK(net_->data_provider()) << "To use rank ohem, data provider must be provided";
		
		seq_rank_accessor_.reset(new data_model::sequential_rank_accessor(*net_->data_provider()));
		seq_rank_accessor_->init();

		original_accessor_ = net_->data_provider()->get_data_accessor();
		net_->data_provider()->set_data_accessor(seq_rank_accessor_);
		net_->data_provider()->begin_prefetch();
	}

	data_type sgd_solver_rank_ohem::train(int iterations, int running_iterations, std::vector<metric> &metrics) {
		metrics.clear();
		context_->set_phase(caffepro_context::TRAIN);

		bool debug_mode = (context_->get_global_cfg(GLOBALCFGNAME_DEBUG_MODE) == "TRUE");

		data_type global_lr = 0, global_wc = param_.weight_decay();
		double total_fw_time = 0, total_bp_time = 0, total_update_time = 0;
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
			if ((running_iterations + i + 1) % param_.update_interval() == 0) { // LAST batch for an update interval
				global_lr = get_learning_rate(iter_ + i);
				updater_->update(global_lr, global_wc, true);
				context_->sync_all_devices();
			}
			update_rank();
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
		metric sum_t = { "solver", "Sum time", (total_fw_time + total_bp_time + total_update_time) / iterations };
		metric lr = { "solver", "Learning rate", global_lr };
		metrics.push_back(fw_t);
		metrics.push_back(bp_t);
		metrics.push_back(up_t);
		metrics.push_back(sum_t);
		metrics.push_back(lr);

		if (param_.has_train_primary_output_index()) {
			CHECK_LT(param_.train_primary_output_index(), metrics.size());
			return (data_type)metrics[param_.train_primary_output_index()].value;
		}

		return (data_type)metrics[0].value;
	}

	data_type sgd_solver_rank_ohem::test(std::vector<metric> &metrics) {
		CHECK(test_net_);
		metrics.clear();
		bool debug_mode = (context_->get_global_cfg(GLOBALCFGNAME_DEBUG_MODE) == "TRUE");

		// first, eval training error and loss
		context_->set_phase(caffepro_context::TRAIN);
		net_->data_provider()->finish_prefetch();
		net_->data_provider()->set_data_accessor(original_accessor_); // set to original accessor
		net_->data_provider()->begin_prefetch();

		std::vector<metric> training_metrics;
		for (int i = 0; i < param_.test_iter(); i++) {
			if (debug_mode) {
				net_->forward_debug(true);
			}
			else {
				net_->forward(true);
			}
			context_->sync_all_devices();

			merge_metrics(training_metrics, *net_);
		}

		for (int i = 0; i < (int)training_metrics.size(); i++) {
			training_metrics[i].name = "Training " + training_metrics[i].name;
			training_metrics[i].value /= param_.test_iter();
		}

		net_->data_provider()->finish_prefetch();
		net_->data_provider()->set_data_accessor(seq_rank_accessor_); // recover seq rank accessor
		net_->data_provider()->begin_prefetch();
		net_->release_blobs();

		// then, eval test error and loss
		context_->set_phase(caffepro_context::TEST);

		for (int i = 0; i < param_.test_iter(); i++) {
			if (debug_mode) {
				test_net_->forward_debug(true);
			}
			else {
				test_net_->forward(true);
			}
			context_->sync_all_devices();

			merge_metrics(metrics, *test_net_);
		}

		for (int i = 0; i < (int)metrics.size(); i++) {
			metrics[i].value /= param_.test_iter();
		}

		metrics.insert(metrics.end(), training_metrics.begin(), training_metrics.end());

		if (param_.has_test_primary_output_index()) {
			CHECK_LT(param_.test_primary_output_index(), metrics.size());
			return (data_type)metrics[param_.test_primary_output_index()].value;
		}

		return (data_type)metrics[0].value;
	}

	void sgd_solver_rank_ohem::update_rank() {
		// get scores container
		softmax_loss_layer *loss_layer = net_->get_layer<softmax_loss_layer>(net_->layers());
		CHECK(loss_layer) << "Rank ohem needs softmax loss layer";
		auto &scores = *loss_layer->avg_prob();
		
		// get batch data
		auto &cur_batch = net_->data_provider()->current_batch()->batch_data;
		CHECK_EQ(cur_batch.size(), scores.sum_num());

		auto *rank_accessor = dynamic_cast<data_model::sequential_rank_accessor *>(
			net_->data_provider()->get_data_accessor().get());

		CHECK(rank_accessor);


		// get rank for each instance
		int num = scores.sum_num();
		CHECK(scores.same_inner_count());
		int feature_dim = scores.get(0)->inner_count();

		for (int n = 0; n < num; n++) {
			const data_type *score = scores.get_cpu_data_across_dev(n);
			data_type rank = -FLT_MAX;
			for (int d = 0; d < feature_dim; d++) {
				rank = std::max(rank, score[d]);
			}
			rank = 1 - rank; // higher score will result in lower rank

			const string &data_name = cur_batch[n].original_data->data_name;
			rank_accessor->update_score(data_name, rank);
		}
	}
}