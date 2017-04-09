
#include <caffepro/test_time/net_tester.h>
#include <caffepro/layers/softmax_loss_layer.h>
#include <caffepro/utils/analyzer.h>
#include <caffepro/proto/proto_io.h>

namespace caffepro {
	using std::string;

	net_tester::net_tester(boost::shared_ptr<caffepro_net> net) 
		: net_(net) {
		// nothing to do
	}

	void net_tester::run(int num_iters, int display_iters, const string &file_info) {
		CHECK_GE(num_iters, 0);
		CHECK_GT(display_iters, 0);

		std::vector<metric> metrics, partial_metrics;
		net_->context()->set_phase(caffepro_context::TEST);
		int img_count = 0;
		auto &img_info = net_->data_provider()->test_img_info();
		img_info->Clear();
		for (int iter = 0; iter < num_iters; iter++) {
			net_->forward(true);
			net_->context()->sync_all_devices();

			string layer_name_ = "loss";
			auto layer = net_->get_layer(layer_name_);
			CHECK_GT(layer->inputs().size(), 0);
			node_blob *source = layer->inputs()[0].get();
			if (layer->layer_param().type() == "softmax_loss") {
				source = dynamic_cast<softmax_loss_layer &>(*layer.get()).prob().get();
			}
			int feature_dim = source->get(0)->inner_count();

			for (int k = 0; k < source->sum_num(); k++) {
				const data_type *data = source->get_cpu_data_across_dev(k);
				data_type max_v = -1000000;
				int max_index = -1;
				for (int j = 0; j < feature_dim; j++) {
					if (data[j] > max_v) {
						max_v = data[j];
						max_index = j;
					}
					img_info->mutable_images(img_count)->add_prob(data[j]);
				}
				img_info->mutable_images(img_count)->set_answer(max_index);
				img_count++;
			}

			merge_metrics(metrics, *net_);
			merge_metrics(partial_metrics, *net_);

			if ((iter + 1) % display_iters == 0) {
				for (auto &v : partial_metrics) {
					v.value /= display_iters;
				}
				display_metrics(iter + 1, partial_metrics, "TEST");
				partial_metrics.clear();
			}

			// invoke hooks
			/*hook_handler_args hook_args;
			hook_args.content = net_.get();
			net_->context()->hooks()->invoke(HOOK_AFTER_NET_FORWARD, *this, hook_args);*/
		}


		// DeepTracker-test: manual test function, dump .info to dist
		analyzer::DumpInfo imgInfos;
		if (!file_info.empty()) {
			analyzer_proto::Info info;
			proto_io(info).from_binary_file(file_info);
			img_info->set_iteration(info.iteration());
			imgInfos.testRecord(*img_info, info.iteration(), "output");
		}
		else {
			img_info->set_iteration(0);
			imgInfos.testRecord(*img_info, 0, "output", "0");
		}
		img_info->Clear();

		for (auto &v : metrics) {
			v.value /= num_iters;
		}
		display_metrics(num_iters, metrics, "Total");
	}

	void net_tester::merge_metrics(std::vector<metric> &metrics, caffepro_net &net) {
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

	void net_tester::display_metrics(int cur_iter, std::vector<metric> &metrics, const std::string prefix) {
		LOG(ERROR) << "--" << prefix << ": " << "iter = " << cur_iter;
		for (int i = 0; i < (int)metrics.size(); i++) {
			LOG(ERROR) << "(" << metrics[i].source << ") " << metrics[i].name << ": " << metrics[i].value;
		}
		LOG(ERROR) << "";
		::google::FlushLogFiles(0);
	}
}