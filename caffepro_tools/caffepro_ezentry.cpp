
#include "caffepro_ezentry.h"

#include <caffepro/utils/filesystem.h>
#include <caffepro/solver/sgd_solver.h>
#include <caffepro/solver/sgd_solver_ohem.h>
#include <caffepro/solver/sgd_solver_ohem_fast.h>
#include <caffepro/solver/sgd_solver_benchmark.h>
#include <caffepro/solver/sgd_solver_rank_ohem.h>
#include <caffepro/test_time/bn_postprocess.h>
#include <caffepro/test_time/net_tester.h>
#include <caffepro/test_time/net_test_extract.h>
#include <caffepro/hooks/save_feature_handler.h>
#include <caffepro/hooks/hook_triggers.h>

#include <caffepro/proto/proto_io.h>

namespace caffepro_tools {

	using namespace caffepro;

	inline void init_log() {
		if (!filesystem::exist("./log")) {
			filesystem::create_directory("./log");
		}
		::google::SetLogDestination(0, "./log/");
	}

	void train_ohem(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &protobin_file, bool finetune, const std::string &updater) {
		init_log();

		sgd_solver_ohem solver(context, prototxt_file, nullptr, nullptr, updater);

		if (!protobin_file.empty()) {
			if (finetune) {
				solver.load_net_param_only(protobin_file);
			}
			else {
				solver.load(protobin_file);
			}
		}

		solver.run();
	}

	void train_ohem_fast(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &protobin_file, bool finetune, const std::string &updater) {
		init_log();

		sgd_solver_ohem_fast solver(context, prototxt_file, nullptr, nullptr, updater);

		if (!protobin_file.empty()) {
			if (finetune) {
				solver.load_net_param_only(protobin_file);
			}
			else {
				solver.load(protobin_file);
			}
		}

		solver.run();
	}

	void train_rank_ohem(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &protobin_file, bool finetune, const std::string &updater) {
		init_log();

		sgd_solver_rank_ohem solver(context, prototxt_file, nullptr, nullptr, updater);

		if (!protobin_file.empty()) {
			if (finetune) {
				solver.load_net_param_only(protobin_file);
			}
			else {
				solver.load(protobin_file);
			}
		}

		solver.run();
	}

	void train(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &protobin_file, bool finetune, const std::string &updater) {
		init_log();

		// DeepTracker-4: initialize solver to start the training process
		sgd_solver solver(context, prototxt_file, nullptr, nullptr, updater);

		if (!protobin_file.empty()) {
			if (finetune) {
				solver.load_net_param_only(protobin_file);
			}
			else {
				solver.load(protobin_file);
			}
		}

		solver.run();
	}

	void extract_feature(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &model_file, int nIters, const std::string &layer_) {
		init_log();

		NetParameter net_param;
		proto_io(net_param).from_text_file(prototxt_file);
		boost::shared_ptr<caffepro_net> net(caffepro_net::create_from_proto(context, net_param));

		NetParameter net_weights;
		proto_io(net_weights).from_binary_file(model_file);
		net->load_weights(net_weights);

		net_test_extract extractor(net);
		extractor.run(nIters, layer_);
	}

	void benchmark(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &model_file, bool show_details) {
		init_log();

		sgd_solver_benchmark solver(context, prototxt_file);

		if (!model_file.empty()) {
			solver.load_net_param_only(model_file);
		}
		solver.benchmark(show_details);
	}

	void bn_post_process(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &model_file, int nIters,
		const std::string &save_file) {

		init_log();

		bn_postprocess processor(context, prototxt_file, model_file);
		processor.run(nIters);
		// processor.save(save_file);

		context->events()->wait_all();
	}

	void bn_post_process(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &model_file, int nIters, 
		const std::string &save_file, const analyzer_proto::Info &info) {

		init_log();

		bn_postprocess processor(context, prototxt_file, model_file, info);
		processor.run(nIters);
		processor.save(save_file);

		context->events()->wait_all();
	}

	
	// DeepTracker-test: manual test function
	void test(caffepro::caffepro_context *context, const std::string &prototxt_file, const std::string &model_file, int nIters, int display_iters, const std::string &info_file) {
		
		init_log();

		NetParameter net_param;
		proto_io(net_param).from_text_file(prototxt_file);

		// modify BN attribute
		bool found_bn = false;
		for (int i = 0; i < (int)net_param.layers_size(); i++) {
			auto &layer_param = *net_param.mutable_layers(i)->mutable_layer();

			if (layer_param.type() == "batch_norm" || layer_param.type() == "mean_norm") {
				if (!found_bn) {
					found_bn = true;
					LOG(ERROR) << "BN layer detected. BN type will be set to USE_RECORD_NORM by default";
				}
				layer_param.mutable_batch_norm_param()->set_record_option(BatchNormalizationParameter_RecordOption_USE_RECORD_NORM);
			}
		}

		boost::shared_ptr<caffepro_net> net(caffepro_net::create_from_proto(context, net_param));

		NetParameter net_weights;
		proto_io(net_weights).from_binary_file(model_file);
		net->load_weights(net_weights);  // load weight parameters

		net_tester tester(net);
		tester.run(nIters, display_iters, info_file);

		context->events()->wait_all();
	}
}