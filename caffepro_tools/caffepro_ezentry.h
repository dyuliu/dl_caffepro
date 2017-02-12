
#pragma once 

#include <caffepro/context/caffepro_context.h>
#include <caffepro/utils/analyzer.h>
#include <string>

namespace caffepro_tools {
	void train(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &protobin_file,
		bool finetune = false,
		const std::string &updater = "SGD"
		);

	void train_ohem(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &protobin_file,
		bool finetune = false,
		const std::string &updater = "SGD"
		);

	void train_ohem_fast(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &protobin_file,
		bool finetune = false,
		const std::string &updater = "SGD"
		);

	void train_rank_ohem(
		caffepro::caffepro_context *context, 
		const std::string &prototxt_file, 
		const std::string &protobin_file, 
		bool finetune = false,
		const std::string &updater = "SGD"
		);

	void extract_feature(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &model_file,
		int nIters,
		const std::string &layer_
		);

	void benchmark(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &model_file,
		bool show_details
		);

	void bn_post_process(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &model_file,
		int nIters, 
		const std::string &save_file
		);

	void bn_post_process(
		caffepro::caffepro_context *context,
		const std::string &prototxt_file,
		const std::string &model_file,
		int nIters,
		const std::string &save_file,
		const analyzer::Info &info_file
		);

	void box_tester(
		const std::string &net_proto_txt,
		const std::string &net_proto_bin,
		const std::string &save_feature_file_name,
		bool full_result
		);

	void test(caffepro::caffepro_context *context, 
		const std::string &prototxt_file, 
		const std::string &model_file, 
		int nIters, 
		int display_iters,
		const std::string &info_file = ""
		);
}