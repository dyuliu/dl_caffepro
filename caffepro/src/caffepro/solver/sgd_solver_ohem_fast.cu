

#include <caffepro/layers/softmax_ohem_layer.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/string_uitls.h>
#include <caffepro/updater/updaters.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/solver/sgd_solver_ohem_fast.h>

#include <iostream>

namespace caffepro {
	using std::vector;
	using std::string;
	using std::cout;
	using std::endl;

	sgd_solver_ohem_fast::sgd_solver_ohem_fast(caffepro_context *context, const std::string &solver_param_file,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: sgd_solver(context, solver_param_file, dataprovider_train, dataprovider_test, default_updater) {
	}

	sgd_solver_ohem_fast::sgd_solver_ohem_fast(caffepro_context *context, SolverParameter &solver_param,
		boost::shared_ptr<data_model::data_provider> dataprovider_train, boost::shared_ptr<data_model::data_provider> dataprovider_test,
		const string &default_updater)
		: sgd_solver(context, solver_param, dataprovider_train, dataprovider_test, default_updater) {
	}

	__global__ static void select(const int n, const int item_size,
		const data_type *mapper,
		const data_type *input_data,
		data_type *output_data) {

		CUDA_KERNEL_LOOP(index, n) {

			const int ins = index / item_size;
			const int j = index % item_size;

			const int i = (int)floorf(mapper[ins] - 1 + .5);

			if (i >= 0) output_data[i * item_size + j] = input_data[index];
		}
	}

	void sgd_solver_ohem_fast::select_by_indicators(boost::shared_ptr<node_blob> input,
		const boost::shared_ptr<node_blob>& bp_indicator, const int ohem_size) {

		string blob_name = input->get_name();
		//if (input->get(0)->num() == ohem_size / input->size()) return;

		// select output data

		const int num_device = bp_indicator->size();

		//CHECK_EQ(bp_indicator->size(), num_device);

		boost::shared_ptr<node_blob> output;

		output.reset(new node_blob());

		const int ohem_size_per_gpu = ohem_size / num_device;

		if (input->size() == num_device) {

			for (int nd = 0; nd < num_device; nd++) {

				if (ohem_size_per_gpu == input->get(nd)->num())
					return;

				output->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
					input->get(nd)->context(),
					ohem_size_per_gpu,
					input->get(nd)->channels(),
					input->get(nd)->height(),
					input->get(nd)->width(),
					input->get(nd)->device_id()
					)
					));
			}

			for (int nd = 0; nd < num_device; nd++) {
				ENTER_DEVICE_CONTEXT(input->get(nd)->device_id())

					cublas_wrapper<data_type> cublas(input->get(nd)->context(), input->get(nd)->device_id());

				CHECK_EQ(bp_indicator->get(nd)->num(), input->get(nd)->num());
				const int item_size = input->get(nd)->channels() * input->get(nd)->height() * input->get(nd)->width();

				const data_type* input_data = input->get(nd)->gpu_data();
				data_type* selected_data = output->get(nd)->mutable_gpu_data();

				const int count = input->get(nd)->count();
				KERNEL_CALL(select, count)(
					count,
					item_size,
					bp_indicator->get(nd)->gpu_data(),
					input_data,
					selected_data
					);
				CUDA_POST_KERNEL_CHECK;

				//const data_type* bp_indicator_data = bp_indicator->get(nd)->cpu_data();
				//for (int i = 0; i < bp_indicator->get(nd)->num(); i++) {
				//	if (bp_indicator_data[i] > 0) {
				//		//memcpy(selected_data, input_data, item_size * sizeof(data_type));
				//		//selected_data += item_size;

				//		memcpy(selected_data + (int)(bp_indicator_data[i] - 1 + .5f) * item_size, input_data, item_size * sizeof(data_type));
				//		//cublas.copy(item_size, input_data, selected_data);
				//		
				//		
				//	}
				//	input_data += item_size;
				//}

				//for (int i = 0; i < ohem_size_per_gpu; i++) {
				//	{
				//		memcpy(selected_data, input_data, item_size * sizeof(data_type));
				//		//cublas.copy(item_size, input_data, selected_data);
				//		selected_data += item_size;
				//	}
				//	input_data += item_size;
				//}

				// copy back
				input->get(nd)->reshape_like(*output->get(nd));
				cublas.copy(output->get(nd)->count(), output->get(nd)->gpu_data(), input->get(nd)->mutable_gpu_data());

				//memset(output->get(nd)->mutable_cpu_data(), 0, output->get(nd)->count() * sizeof(data_type));
				//memcpy(input->get(nd)->mutable_cpu_data(), output->get(nd)->cpu_data(), output->get(nd)->count() * sizeof(data_type));

				EXIT_DEVICE_CONTEXT;
			}
		}
		else if (input->size() == 1) {

			//return;
			//NOT_IMPLEMENTED;
			std::vector<data_type>  bp_indicator_concat;
			bp_indicator_concat.resize(0);

			for (int nd = 0; nd < bp_indicator->size(); nd++) {
				const data_type *bp_indicator_data = bp_indicator->get(nd)->cpu_data();
				for (int i = 0; i < bp_indicator->get(nd)->num(); i++) {
					bp_indicator_concat.push_back((bp_indicator_data[i] > 0) ? (bp_indicator_data[i] + nd * ohem_size_per_gpu) : 0);
				}
			}

			output->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
				input->get(0)->context(),
				ohem_size,
				input->get(0)->channels(),
				input->get(0)->height(),
				input->get(0)->width(),
				input->get(0)->device_id()
				)
				));

			cublas_wrapper<data_type> cublas(input->get(0)->context(), input->get(0)->device_id());

			CHECK_EQ(bp_indicator_concat.size(), input->get(0)->num());
			const int item_size = input->get(0)->channels() * input->get(0)->height() * input->get(0)->width();
			const data_type* input_data = input->get(0)->gpu_data();
			data_type* selected_data = output->get(0)->mutable_gpu_data();
			for (int i = 0; i < bp_indicator_concat.size(); i++) {
				if (bp_indicator_concat[i] > 0) {
					//memcpy(selected_data, input_data, item_size * sizeof(data_type));
					//selected_data += item_size;

					//memcpy(selected_data + (int)(bp_indicator_concat[i] - 1 + .5f) * item_size, input_data, item_size * sizeof(data_type));
					cublas.copy(item_size, input_data, selected_data + (int)floorf(bp_indicator_concat[i] - 1 + .5f) * item_size);

				}
				input_data += item_size;
			}

			// copy back
			input->get(0)->reshape_like(*output->get(0));
			cublas.copy(output->get(0)->count(), output->get(0)->gpu_data(), input->get(0)->mutable_gpu_data());
			//memcpy(input->get(0)->mutable_cpu_data(), output->get(0)->cpu_data(), output->get(0)->count() * sizeof(data_type));
		}
		else {
			LOG(FATAL) << "select_by_indicators error.";
		}

	}

	data_type sgd_solver_ohem_fast::train(int iterations, int running_iterations, std::vector<metric> &metrics) {
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
				net_->forward(false);

				CHECK_EQ(net_->layers()[0]->layer_param().type(), "data_bigfile");

				// preparing copy
				std::vector<boost::shared_ptr<node_blob>>& net_output_blobs = net_->output_blobs();
				std::vector<boost::shared_ptr<node_blob>>& data_layer_blobs = net_->layers()[0]->outputs();

				CHECK_EQ(net_->layers().back()->layer_param().type(), "softmax_ohem_split");
				CHECK_EQ(2, net_output_blobs.size());

				// remember the original nums			
				num_total = data_layer_blobs[1]->get(0)->num();
				nums.resize(data_layer_blobs[0]->size());
				for (int nd = 0; nd < data_layer_blobs[0]->size(); nd++)
					nums[nd] = data_layer_blobs[0]->get(nd)->num();

				// reorder
				const boost::shared_ptr<node_blob>& bp_indicator = net_output_blobs[1];

				int ohem_size = 0;
				for (int nd = 0; nd < bp_indicator->size(); nd++)
				for (int i = 0; i < bp_indicator->get(nd)->num(); i++)
					ohem_size += (bp_indicator->get(nd)->cpu_data()[i] > 0) ? 1 : 0;


				select_by_indicators(net_->layers()[0]->outputs()[0], bp_indicator, ohem_size);
				select_by_indicators(net_->layers()[0]->outputs()[1], bp_indicator, ohem_size);

				int idx_gpuconcat = 1;
				bool gpuconcat = false;
				for (int j = 1; j < net_->layers().size() - 1; j++, idx_gpuconcat++) {

					string layer_name = net_->layers()[j]->layer_param().name();

					std::vector<boost::shared_ptr<node_blob>>& current_blob = net_->layers()[j]->outputs();

					for (int k = 0; k < current_blob.size(); k++) {
						string blob_name = current_blob[k]->get_name();

						if (current_blob[k]->size() == 1 && bp_indicator->size() > 1) {
							gpuconcat = true;
							break;
						}
						select_by_indicators(current_blob[k], bp_indicator, ohem_size);
					}
					if (gpuconcat) break;
				}

				for (int j = 0; j < net_->layers().size(); j++) {
					string layer_name = net_->layers()[j]->layer_param().name();
					net_->layers()[j]->resize();
				}

				// recompute last layer
				const string name_start_layer = net_->layers()[idx_gpuconcat]->layer_param().name();
				//const string name_start_layer = net_->layers()[0+1]->layer_param().name();
				const string name_end_layer = net_->layers()[net_->layers().size() - 1]->layer_param().name();
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