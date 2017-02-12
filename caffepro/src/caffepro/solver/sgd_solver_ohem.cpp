
#include <caffepro/solver/sgd_solver_ohem.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/string_uitls.h>
#include <caffepro/updater/updaters.h>

#include <iostream>

namespace caffepro {
	using std::vector;
	using std::string;
	using std::cout;
	using std::endl;

	sgd_solver_ohem::sgd_solver_ohem(caffepro_context *context, const std::string &solver_param_file,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: sgd_solver(context, solver_param_file, dataprovider_train, dataprovider_test, default_updater) {
	}

	sgd_solver_ohem::sgd_solver_ohem(caffepro_context *context, SolverParameter &solver_param,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: sgd_solver(context, solver_param, dataprovider_train, dataprovider_test, default_updater) {
	}

	data_type sgd_solver_ohem::train(int iterations, int running_iterations, std::vector<metric> &metrics) {
		metrics.clear();
		context_->set_phase(caffepro_context::TRAIN);

		bool debug_mode = (context_->get_global_cfg(GLOBALCFGNAME_DEBUG_MODE) == "TRUE");

		data_type global_lr = 0, global_wc = param_.weight_decay();
		double total_fw_time = 0, total_bp_time = 0, total_update_time = 0;
		for (int i = 0; i < iterations; i++) {
			clock_t start_time = clock();
			
			std::vector<int> nums;
			int num_total;

			if (debug_mode) {
				NOT_IMPLEMENTED;
				//net_->forward_debug(false);
			}
			else {
				net_->forward(true);

				CHECK_EQ(net_->layers()[0]->layer_param().type(), "data_bigfile");

				// preparing copy
				std::vector<boost::shared_ptr<node_blob>>& net_output_blobs = net_->output_blobs();
				std::vector<boost::shared_ptr<node_blob>>& data_layer_blobs = net_->layers()[0]->outputs();

				CHECK_EQ(net_->layers().back()->layer_param().type(), "softmax_ohem");
				CHECK_GE(net_output_blobs.size(), 3);

				const int width = net_output_blobs[2]->get(0)->width();
				const int height = net_output_blobs[2]->get(0)->height();
				const int channels = net_output_blobs[2]->get(0)->channels();
				CHECK_EQ(width, data_layer_blobs[0]->dim_at(0));
				CHECK_EQ(height, data_layer_blobs[0]->dim_at(1));
				CHECK_EQ(channels, data_layer_blobs[0]->dim_at(2));

				CHECK_EQ(1, net_output_blobs[1]->dim_at(2)); //label
				CHECK_EQ(1, data_layer_blobs[1]->dim_at(2)); //label

				// remember the original nums			
				num_total = data_layer_blobs[1]->get(0)->num();
				nums.resize(data_layer_blobs[0]->size());
				for (int nd = 0; nd < data_layer_blobs[0]->size(); nd++)
					nums[nd] = data_layer_blobs[0]->get(nd)->num();

				// copy
				data_layer_blobs[1]->get(0)->reshape_4d(
					net_output_blobs[1]->get(0)->num(),
					1,
					1,
					1
					);
				const int count = data_layer_blobs[1]->get(0)->count();
				CHECK_EQ(count, net_output_blobs[1]->get(0)->count());
				memcpy(data_layer_blobs[1]->get(0)->mutable_cpu_data(), net_output_blobs[1]->get(0)->cpu_data(), count * sizeof(data_type));

				//const data_type* selected_label = data_layer_blobs[1]->get(0)->cpu_data();
				//for (int i = 0; i < data_layer_blobs[1]->get(0)->num(); i++)
				//	printf("i: %d, selected_label: %f\n", i, selected_label[i]);
				//printf("\n");

				for (int nd = 0; nd < data_layer_blobs[0]->size(); nd++) {
					data_layer_blobs[0]->get(nd)->reshape_4d(
						net_output_blobs[2]->get(nd)->num(),
						net_output_blobs[2]->get(nd)->channels(),
						net_output_blobs[2]->get(nd)->height(),
						net_output_blobs[2]->get(nd)->width()
						);

					const int count = data_layer_blobs[0]->get(nd)->count();
					CHECK_EQ(count, net_output_blobs[2]->get(nd)->count());
					memcpy(data_layer_blobs[0]->get(nd)->mutable_cpu_data(), net_output_blobs[2]->get(nd)->cpu_data(), count * sizeof(data_type));
				}

				// second pass forward
				const string name_start_layer = net_->layers()[0+1]->layer_param().name();
				const string name_end_layer = net_->layers()[net_->layers().size()-1]->layer_param().name();
				net_->forward_range(name_start_layer, name_end_layer);
			}

			context_->sync_all_devices();
			int forward_time = clock() - start_time;

			start_time = clock();
			bool lazy_update = ((running_iterations + i) % param_.update_interval() != 0); // not the FIRST batch for an update interval
			if (debug_mode) {
				NOT_IMPLEMENTED;
				//net_->backward_debug(true, lazy_update);
			}
			else {
				net_->backward(true, lazy_update);

				//restore size
				std::vector<boost::shared_ptr<node_blob>>& data_layer_blobs = net_->layers()[0]->outputs();

				data_layer_blobs[1]->get(0)->reshape_4d(
					num_total,
					1,
					1,
					1
					);
				
				for (int nd = 0; nd < data_layer_blobs[0]->size(); nd++) {
					data_layer_blobs[0]->get(nd)->reshape_4d(
						nums[nd],
						data_layer_blobs[0]->get(nd)->channels(),
						data_layer_blobs[0]->get(nd)->height(),
						data_layer_blobs[0]->get(nd)->width()
						);
				}
			}
			context_->sync_all_devices();
			int backward_time = clock() - start_time;

			start_time = clock();
			if ((running_iterations + i + 1) % param_.update_interval() == 0) { // LAST batch for an update interval
				global_lr = get_learning_rate(iter_ + i);
				updater_->update(global_lr, global_wc, true);
				context_->sync_all_devices();
			}
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
}