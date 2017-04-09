
#include <gflags.h>
#include <glog/logging.h>
#include <caffepro\utils\filesystem.h>
#include <iostream>
#include <string>
#include <regex>
#include <caffepro/proto/proto_io.h>
#include <caffepro/utils/analyzer.h>

DEFINE_bool(find_conv_algo, true, "Find fastest algorithm for convolution");
DEFINE_int32(top_n, 0, "Show top n accuracy");

DEFINE_string(action, "", "Operations available: train, train_rank_ohem, benchmark, bn_postprocess, box_test, test");
DEFINE_string(prototxt, "", "Define a net or a solver");
DEFINE_string(solverstate, "", "Solver state binary file");
DEFINE_string(model, "", "Model file");
DEFINE_string(info, "", "Info file");
DEFINE_string(save_model, "", "Save model");
DEFINE_bool(show_details, true, "Show details");
DEFINE_int32(iters, 0, "Total iterations");
DEFINE_bool(full_results, false, "Save full results");
DEFINE_bool(all, false, "Dump all info");
DEFINE_string(save_feature, "", "Prefix of save feature file");
DEFINE_int32(display_iters, 100, "Display iterations");
DEFINE_bool(debug_mode, false, "Enable debug mode");
DEFINE_string(updater, "sgd", "Updater type");
DEFINE_string(src, "", "file/folder path");
DEFINE_string(layer, "", "layer name");
DEFINE_string(utils, "", "big2imgs, imgs2big, imgs2mean, extract_feature");

#include <caffepro/context/caffepro_context.h>
#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/caffepro.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/context/common_names.h>
#include "caffepro_ezentry.h"
#include "utils/utils.h"
#include <caffepro/utils/multinode.h>

#include <Windows.h>
#include <omp.h>

using std::cout;
using std::endl;
using std::string;

caffepro::caffepro_context context;

bool init_context() {

	COUT_Y("\n[ ------ Global Configs ------ ]") << endl;

	if (FLAGS_find_conv_algo) {
		context.set_global_cfg(caffepro::GLOBALCFGNAME_FIND_CONV_ALGO, "TRUE");
		COUT_METD << "FIND_CONV_ALGO: " << "TRUE" << endl;
	}
	else {
		COUT_METD << "FIND_CONV_ALGO: " << "FALSE" << endl;
	}

	if (FLAGS_top_n >= 1) {
		context.set_global_cfg(caffepro::GLOBALCFGNAME_SHOW_TOP_N_ACC, std::to_string(FLAGS_top_n));
		COUT_METD << "SHOW_TOP_N_ACC: " << FLAGS_top_n << endl;
	}

	if (FLAGS_debug_mode) {
		context.set_global_cfg(caffepro::GLOBALCFGNAME_DEBUG_MODE, "TRUE");
		COUT_METD << "DEBUG_MODE: ENABLED" << endl;
	}
	
	COUT_Y("\n[ ------ Start to running ------ ]") << endl ;

	return true;
}

bool init_context(caffepro::caffepro_context &context) {
	COUT_Y("\n[ ------ Global Configs ------ ]") << endl;

	if (FLAGS_find_conv_algo) {
		context.set_global_cfg(caffepro::GLOBALCFGNAME_FIND_CONV_ALGO, "TRUE");
		COUT_METD << "FIND_CONV_ALGO: " << "TRUE" << endl;
	}
	else {
		COUT_METD << "FIND_CONV_ALGO: " << "FALSE" << endl;
	}

	if (FLAGS_top_n >= 1) {
		context.set_global_cfg(caffepro::GLOBALCFGNAME_SHOW_TOP_N_ACC, std::to_string(FLAGS_top_n));
		COUT_METD << "SHOW_TOP_N_ACC: " << FLAGS_top_n << endl;
	}

	if (FLAGS_debug_mode) {
		context.set_global_cfg(caffepro::GLOBALCFGNAME_DEBUG_MODE, "TRUE");
		COUT_METD << "DEBUG_MODE: ENABLED" << endl;
	}

	COUT_Y("\n[ ------ Start to running ------ ]") << endl;

	return true;
}

void MPI_PingPong(caffepro::multinode &instance_) {

	double last_dur_1 = 0, last_dur_2 = 0, last_n = 0, n = 0, symbol = -1;
	double alpha = 0, gamma = 0, beta = 0, combine = 0, predict = 0;
	double p = instance_.get_worker_size();
	double m = 0, n_v = 0;

	for (int count = 0; count <= 8; count++) {
		for (int rep = 0; rep < 5; rep++) {
			int i = 0;
			if (count % 2 == 0) {
				i = FLAGS_iters*(count + 1);
				n = count+1;
				m = i / 1000000.0;
			}
			else {
				i = FLAGS_iters*(count + 2);
				n = count+2;
				m = i / 1000000.0;
			}

			float *sum = new float[i];

			instance_.barrier();
			auto start_2 = instance_.now();
			instance_.all_sum(sum, i);
			instance_.barrier();
			auto end_2 = instance_.now();

			auto dur_2 = (end_2 - start_2) * 1000;

			if (last_n != 0) {
				double k = double(n) / last_n;
				alpha = (k*last_dur_2 - dur_2) / (2 * (k - 1)*log2(p));
				combine = (dur_2 - 2 * log2(p) * alpha)*p / ((p - 1) * n);
				predict = (2 * log2(p)*alpha) + (p - 1)*n*combine / p;
				//beta = (dur_2 - 2 * alpha*log2(p) - (p - 1)*n*gamma / p)*p / (2 * (p - 1));
			}

			COUT_WORKID(p) << "Count: " << count
				//<< "MPI reduce_scatter, size: " << i << ", time: " << (end_1 - start_1) * 1000 
				<< ", MPI Allreuce, size: " << i << ", time: " << dur_2
				<< ", combine: " << combine << ", alpha: " << alpha
				<< ", predict: " << predict << std::endl;

			n_v += dur_2 / m;

			//last_dur_1 = dur_1;
			last_dur_2 = dur_2;
			last_n = n;

			delete[] sum;
		}
	}

	COUT_WORKID(p) << "avr: " << n_v / 30.0 << std::endl;

}

void train(int argc, char *argv[]) {

	caffepro::multinode* instance_ = caffepro::multinode::get();
	instance_->init(argc, argv);
	// MPI test
	data_type sum_ = (data_type)instance_->get_worker_id();
	instance_->all_sum(&sum_, 1);
	// GPU test
	caffepro::gpu_info(instance_->get_worker_id());
	// MPI PingPong
	//if (instance_->get_worker_size() > 1)
	//	MPI_PingPong(*instance_);

	// check if the solverstate exists
	if (FLAGS_solverstate.empty()) {
		auto solverstae_list = caffepro::filesystem::get_files("./", std::string("*"+std::to_string(instance_->get_worker_id())+".solverstate").c_str(), false);
		std::smatch match;
		int max_iter = std::numeric_limits<int>::min();
		std::string max_file = "";
		for (auto i : solverstae_list)  {
			if (std::regex_search(i, match, std::regex("_iter_(.*)_id"))) {
				if (max_iter < atoi(match[1].str().c_str())) {
					max_iter = atof(match[1].str().c_str());
					max_file = i;
				}
			}
		}
		if (solverstae_list.size()) {
			COUT_CHEK << "Load solverstate file from " << max_file << std::endl;
			FLAGS_solverstate = max_file;
		}
	}

	COUT_WORKID(instance_->get_worker_id()) << "Sum of rank value is " << sum_ << std::endl;

	// DeepTracker-2: three types for starting a training process
	if (!FLAGS_prototxt.empty()) {
		if (!FLAGS_solverstate.empty()) {
			COUT_READ << "Continue training from " << FLAGS_solverstate << endl;
			COUT_READ << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train(&context, FLAGS_prototxt, FLAGS_solverstate, false, FLAGS_updater);
		}
		else if (!FLAGS_model.empty()) {
			COUT_READ << "Finetune from " << FLAGS_model << endl;
			COUT_READ << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train(&context, FLAGS_prototxt, FLAGS_model, true, FLAGS_updater);
		} 
		else {
			COUT_READ << "Training from scratch" << endl;
			COUT_READ << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train(&context, FLAGS_prototxt, "", false, FLAGS_updater);  // DeepTracker-3: we mainly use this one
		}
	}
	else {
		COUT_WARN << "Missing prototxt file" << endl;
	}

	COUT_SUCC << "Close the multinode system" << endl;
}

///---------------------------------------------------------------------------------------
void utils() {
	if (FLAGS_utils == "big2imgs") {
		caffepro::big2imgs_batch(FLAGS_src);
	}
	else if (FLAGS_utils == "imgs2big") {
		caffepro::imgs2big_batch(FLAGS_src);
	}
	else if (FLAGS_utils == "imgs2mean") {
		caffepro::imgs2mean(FLAGS_src);
	}
	else if (FLAGS_utils == "extract_model") {
		caffepro::extract_model(FLAGS_prototxt, FLAGS_model, false);
	}
	else if (FLAGS_utils == "extract_grad") {
		caffepro::extract_grad(FLAGS_prototxt, FLAGS_model, false);
	}
	else if (FLAGS_utils == "extract_model_folder") {
		caffepro::extract_model(FLAGS_prototxt, FLAGS_model, true);
	}
	else if (FLAGS_utils == "extract_grad_folder") {
		caffepro::extract_grad(FLAGS_prototxt, FLAGS_model, true);
	}
	else if (FLAGS_utils == "extract_feature") {
		COUT_RUNN << "Task committed: EXTRACT FEATURE" << endl;
		COUT_READ << "Finetune from " << FLAGS_model << endl;
		COUT_READ << "Prototxt file = " << FLAGS_prototxt << endl;
		COUT_RUNN << "Updater = " << FLAGS_updater << endl;
		caffepro_tools::extract_feature(&context, FLAGS_prototxt, FLAGS_model, FLAGS_iters, FLAGS_layer);
	}
	else {
		COUT_WARN << "The cmd is wrong, please check it." << std::endl;
	}
	COUT_SUCC << "Finished all jobs" << endl;
}

///---------------------------------------------------------------------------------------

void benchmark() {
	if (!FLAGS_prototxt.empty()) {
		if (!FLAGS_model.empty()) {
			COUT_RUNN << "Task committed: " << endl;
			COUT_RUNN << "Benchmarking from " << FLAGS_model << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Show details = " << FLAGS_show_details << endl;
			caffepro_tools::benchmark(&context, FLAGS_prototxt, FLAGS_model, FLAGS_show_details);
		}
		else {
			COUT_RUNN << "Task committed: " << endl;
			COUT_RUNN << "Benchmarking from scratch" << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Show details = " << FLAGS_show_details << endl;
			caffepro_tools::benchmark(&context, FLAGS_prototxt, "", FLAGS_show_details);
		}
	}
	else {
		cout << "Missing prototxt file" << endl;
	}
}

void bn_postprocess() {
	if (FLAGS_prototxt.empty()) {
		COUT_WARN << "Missing prototxt file" << endl;
	}
	else if (FLAGS_model.empty()) {
		COUT_WARN << "Missing model file" << endl;
	}
	else if (FLAGS_save_model.empty()) {
		COUT_WARN << "Missing save model file" << endl;
	}
	else if (FLAGS_iters > 0) {
		COUT_RUNN << "Task committed: " << endl;
		COUT_RUNN << "Run test from " << FLAGS_model << endl;
		COUT_RUNN << "Where: " << endl;
		COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
		COUT_RUNN << "Save file = " << FLAGS_save_model << endl;
		caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, FLAGS_iters, FLAGS_save_model);
	}
	else {
		COUT_WARN << "Missing running iterations" << endl;
	}
}

void box_test() {
	if (!FLAGS_prototxt.empty()) {
		COUT_WARN << "Missing prototxt file" << endl;
	}
	else if (!FLAGS_model.empty()) {
		COUT_WARN << "Missing model file" << endl;
	}
	else if (!FLAGS_save_feature.empty()) {
		COUT_RUNN << "Task committed: " << endl;
		COUT_RUNN << "Test boxes from " << FLAGS_model << endl;
		COUT_RUNN << "Where: " << endl;
		COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
		COUT_RUNN << "Save file = " << FLAGS_save_feature << endl;
		COUT_RUNN << "Save full results = " << FLAGS_full_results << endl;

		caffepro_tools::box_tester(FLAGS_prototxt, FLAGS_model, FLAGS_save_feature, FLAGS_full_results);
	}
	else {
		COUT_WARN << "Missing save feature file" << endl;
	}
}

void test() {
	if (FLAGS_prototxt.empty()) {
		COUT_WARN << "Missing prototxt file" << endl;
	}
	else if (FLAGS_model.empty()) {
		COUT_WARN << "Missing model file" << endl;
	}
	else if (FLAGS_iters > 0) {
		COUT_RUNN << "Task committed: " << endl;
		COUT_RUNN << "Batch normalization post-process from " << FLAGS_model << endl;
		COUT_RUNN << "Where: " << endl;
		COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
		COUT_RUNN << "Total iterations = " << FLAGS_iters << endl;
		COUT_RUNN << "Display iterations = " << FLAGS_display_iters << endl;
		if (!FLAGS_save_feature.empty()) {
			COUT_CHEK << "Save feature file = " << FLAGS_save_feature << endl;
		}
		caffepro_tools::test(&context, FLAGS_prototxt, FLAGS_model, FLAGS_iters, FLAGS_display_iters, FLAGS_save_feature);
	}
	else {
		COUT_WARN << "Missing running iterations" << endl;
	}
}

void train_ohem() {
	if (!FLAGS_prototxt.empty()) {
		if (!FLAGS_solverstate.empty()) {
			COUT_RUNN << "OHEM Task committed: " << endl;
			COUT_RUNN << "Continue training from " << FLAGS_solverstate << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_ohem(&context, FLAGS_prototxt, FLAGS_solverstate, false, FLAGS_updater);
		}
		else if (!FLAGS_model.empty()) {
			COUT_RUNN << "OHEM Task committed: " << endl;
			COUT_RUNN << "Finetune from " << FLAGS_model << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_ohem(&context, FLAGS_prototxt, FLAGS_model, true, FLAGS_updater);
		}
		else {
			COUT_RUNN << "OHEM Task committed: " << endl;
			COUT_RUNN << "Training from scratch" << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_ohem(&context, FLAGS_prototxt, "", false, FLAGS_updater);
		}
	}
	else {
		COUT_WARN << "Missing prototxt file" << endl;
	}
}

void train_ohem_fast() {
	if (!FLAGS_prototxt.empty()) {
		if (!FLAGS_solverstate.empty()) {
			COUT_RUNN << "OHEMfast Task committed: " << endl;
			COUT_RUNN << "Continue training from " << FLAGS_solverstate << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_ohem_fast(&context, FLAGS_prototxt, FLAGS_solverstate, false, FLAGS_updater);
		}
		else if (!FLAGS_model.empty()) {
			COUT_RUNN << "OHEMfast Task committed: " << endl;
			COUT_RUNN << "Finetune from " << FLAGS_model << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_ohem_fast(&context, FLAGS_prototxt, FLAGS_model, true, FLAGS_updater);
		}
		else {
			COUT_RUNN << "OHEMfast Task committed: " << endl;
			COUT_RUNN << "Training from scratch" << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_ohem_fast(&context, FLAGS_prototxt, "", false, FLAGS_updater);
		}
	}
	else {
		COUT_WARN << "Missing prototxt file" << endl;
	}
}

void train_rank_ohem() {
	if (!FLAGS_prototxt.empty()) {
		if (!FLAGS_solverstate.empty()) {
			COUT_RUNN << "OHEM Task committed: " << endl;
			COUT_RUNN << "Continue training from " << FLAGS_solverstate << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_rank_ohem(&context, FLAGS_prototxt, FLAGS_solverstate, false, FLAGS_updater);
		}
		else if (!FLAGS_model.empty()) {
			COUT_RUNN << "OHEM Task committed: " << endl;
			COUT_RUNN << "Finetune from " << FLAGS_model << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_rank_ohem(&context, FLAGS_prototxt, FLAGS_model, true, FLAGS_updater);
		}
		else {
			COUT_RUNN << "OHEM Task committed: " << endl;
			COUT_RUNN << "Training from scratch" << endl;
			COUT_RUNN << "Where: " << endl;
			COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
			COUT_RUNN << "Updater = " << FLAGS_updater << endl;
			caffepro_tools::train_rank_ohem(&context, FLAGS_prototxt, "", false, FLAGS_updater);
		}
	}
	else {
		COUT_WARN << "Missing prototxt file" << endl;
	}
}

// -action=testall -model=*.model -prototxt=solver.prototxt -layer = fc10
void testall() {
	// step1 : do bn postprocess
	caffepro::SolverParameter solver_;
	caffepro::proto_io(solver_).from_text_file(FLAGS_prototxt);

	auto train_net_ = solver_.train_net();
	auto test_net_ = solver_.test_net();

	// BN exist judge
	if (FLAGS_model.find("bn_")) {
		FLAGS_iters = 30; // 30-minibatch;
		COUT_CHEK << "BN postprocess iters : " << FLAGS_iters << std::endl;
		FLAGS_prototxt = train_net_;
		FLAGS_save_model = "bn_" + FLAGS_model;
		bn_postprocess();
	}
	else {
		FLAGS_save_model = FLAGS_model;
	}

	COUT_CHEK << "TEST MODEL : " << FLAGS_save_model << std::endl;

	// step2 : test final result
	FLAGS_iters = solver_.test_iter();
	FLAGS_prototxt = test_net_;
	FLAGS_model = FLAGS_save_model;
	test();

	// step3 : extract test features
	// FLAGS_utils = "extract_feature";
	// FLAGS_model = FLAGS_save_model;
	// FLAGS_iters = solver_.test_iter();
	// FLAGS_prototxt = train_net_;
	// utils();
}

void test_dump_model() {
	if (FLAGS_prototxt.empty()) {
		COUT_WARN << "Missing prototxt file, i.g., net_solver.prototxt" << endl;
		return;
	}
	if (FLAGS_info.empty()) {
		COUT_WARN << "Missing info file, e.g., 00002000_000.info or records/" << endl;
		return;
	}
	if (FLAGS_model.empty()) {
		COUT_WARN << "Missing model file, e.g., net_cache_iter_10000_id_0.model" << endl;
		return;
	}
	if (!FLAGS_iters) {
		COUT_WARN << "Please specify the iters for bn_postprocess - 30 for 1P4G 2x" << endl;
		return;
	}
	if (FLAGS_save_model.empty()) {
		FLAGS_save_model = "bn_" + FLAGS_model;
	}

	if (FLAGS_all) {
		auto files = caffepro::filesystem::get_files(FLAGS_info.c_str(), "*.info", false);

		for (int i = 0; i < files.size(); i++) {
			std::cout << files[i];
			caffepro::caffepro_context context;
			init_context(context);
			COUT_CHEK << "Dump test img info: " << files[i] << ", ratio:" << 100 * (i + 1) / float(files.size()) << std::endl;
			// step2: set up solver, load weight from .info, doing bn_postprocess
			caffepro::SolverParameter solver_;
			caffepro::proto_io(solver_).from_text_file(FLAGS_prototxt);
			auto train_net_ = solver_.train_net();
			auto iters = FLAGS_iters;
			auto info_file = files[i];
			analyzer_proto::Info info;
			caffepro::proto_io(info).from_binary_file(info_file);
			std::string save_model = caffepro::fill_zero(info.iteration(), 8);
			caffepro_tools::bn_post_process(&context, train_net_, FLAGS_model, iters, save_model, info);
		}
	}
	else {
		// step2: set up solver, load weight from .info, doing bn_postprocess
		caffepro::SolverParameter solver_;
		caffepro::proto_io(solver_).from_text_file(FLAGS_prototxt);
		FLAGS_prototxt = solver_.train_net();
		auto iters = FLAGS_iters;
		analyzer_proto::Info info;
		caffepro::proto_io(info).from_binary_file(FLAGS_info);
		std::string save_model = caffepro::fill_zero(info.iteration(), 8);
		caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, iters, save_model, info);
	}
}

void test_dump() {
	// step1: check the environment & doing initialization
	if (FLAGS_prototxt.empty()) {
		COUT_WARN << "Missing prototxt file, i.g., net_solver.prototxt" << endl;
		return;
	}
	if (FLAGS_info.empty()) {
		COUT_WARN << "Missing info file, e.g., 00002000_000.info or records/" << endl;
		return;
	}
	if (FLAGS_model.empty()) {
		COUT_WARN << "Missing model file, e.g., net_cache_iter_10000_id_0.model" << endl;
		return;
	}
	if (!FLAGS_iters) {
		COUT_WARN << "Please specify the iters for bn_postprocess - 30 for 1P4G 2x" << endl;
		return;
	}
	if (FLAGS_save_model.empty()) {
		FLAGS_save_model = "bn_" + FLAGS_model;
	}
	
	FLAGS_display_iters = 20;  // default 100

	if (FLAGS_all) {
		auto files = caffepro::filesystem::get_files(FLAGS_info.c_str(), "*.info", false);

		for (int i = 0; i < files.size(); i++) {
			std::cout << files[i];
			caffepro::caffepro_context context;
			init_context(context);
			COUT_CHEK << "Dump test img info: " << files[i] << ", ratio:" << 100 * (i + 1) / float(files.size()) << std::endl;
			// step2: set up solver, load weight from .info, doing bn_postprocess
			caffepro::SolverParameter solver_;
			caffepro::proto_io(solver_).from_text_file(FLAGS_prototxt);
			auto train_net_ = solver_.train_net();
			auto iters = FLAGS_iters;
			auto info_file = files[i];
			COUT_R("BN_POSTPROCESS BEGIN") << endl;
			COUT_RUNN << "Batch normalization post-process from " << FLAGS_model << endl;
			COUT_RUNN << "Prototxt file = " << train_net_ << endl;
			COUT_RUNN << "Save file = " << FLAGS_save_model << endl;
			analyzer_proto::Info info;
			caffepro::proto_io(info).from_binary_file(info_file);
			std::string save_model = caffepro::fill_zero(info.iteration(), 8);
			caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, iters, save_model, info);
			// caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, FLAGS_iters, FLAGS_save_model);

			COUT_R("BN_POSTPROCESS DONE: the model has been saved to " + FLAGS_save_model) << endl;

			// step3: test the model with bn processing
			auto test_net_ = solver_.test_net();
			COUT_R("TEST BEGIN") << endl;
			COUT_RUNN << "Run test from " << FLAGS_save_model << endl;
			COUT_RUNN << "Prototxt file = " << test_net_ << endl;
			COUT_RUNN << "Total iterations = " << solver_.test_iter() << endl;
			COUT_RUNN << "Display iterations = " << FLAGS_display_iters << endl;
			caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, iters, save_model, info);
			COUT_R("TEST DONE");
		}
	}
	else {
		// step2: set up solver, load weight from .info, doing bn_postprocess
		caffepro::SolverParameter solver_;
		caffepro::proto_io(solver_).from_text_file(FLAGS_prototxt);
		FLAGS_prototxt = solver_.train_net();
		auto iters = FLAGS_iters;
		COUT_R("BN_POSTPROCESS BEGIN") << endl;
		COUT_RUNN << "Batch normalization post-process from " << FLAGS_model << endl;
		COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
		COUT_RUNN << "Save file = " << FLAGS_save_model << endl;
		analyzer_proto::Info info;
		caffepro::proto_io(info).from_binary_file(FLAGS_info);
		std::string save_model = caffepro::fill_zero(info.iteration(), 8);
		caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, iters, save_model, info);
		// caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, FLAGS_iters, FLAGS_save_model);

		COUT_R("BN_POSTPROCESS DONE") << endl;

		// step3: test the model with bn processing
		FLAGS_iters = solver_.test_iter();
		FLAGS_prototxt = solver_.test_net();
		FLAGS_model = FLAGS_save_model;
		COUT_R("TEST BEGIN") << endl;
		COUT_RUNN << "Run test from " << FLAGS_model << endl;
		COUT_RUNN << "Prototxt file = " << FLAGS_prototxt << endl;
		COUT_RUNN << "Total iterations = " << FLAGS_iters << endl;
		COUT_RUNN << "Display iterations = " << FLAGS_display_iters << endl;
		caffepro_tools::bn_post_process(&context, FLAGS_prototxt, FLAGS_model, iters, save_model, info);
		COUT_R("TEST DONE");
	}

}

static void failureFunction() { exit(0); }

static BOOL ctrl_handler(DWORD fdwCtrlType) {
	switch (fdwCtrlType) {
	case CTRL_C_EVENT:
		context.set_signal(caffepro::caffepro_context::SIGNAL_STOP_ALL);
		return TRUE;

	default:
		return FALSE;
	}
}

int main(int argc, char *argv[]) {
	::google::InitGoogleLogging(argv[0]);
	::google::InstallFailureFunction(failureFunction);
	omp_set_num_threads(6);
	SetConsoleCtrlHandler((PHANDLER_ROUTINE)ctrl_handler, TRUE);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (!init_context()) {
		return 1;
	}

	// DeepTracker-1: the entry of the tool, we use train action to start a training process
	if (FLAGS_action == "train") {
		train(argc, argv);
	}
	else if (FLAGS_action == "train_ohem") {
		train_ohem();
	}
	else if (FLAGS_action == "train_ohem_fast") {
		train_ohem_fast();
	}
	else if (FLAGS_action == "train_rank_ohem") {
		train_rank_ohem();
	}
	else if (FLAGS_action == "benchmark") {
		benchmark();
	}
	else if (FLAGS_action == "bn_postprocess") {
		bn_postprocess();
	}
	else if (FLAGS_action == "box_test") {
		box_test();
	}
	else if (FLAGS_action == "test") {
		test();
	}
	else if (FLAGS_action == "utils") {
		utils();
	}
	else if (FLAGS_action == "testall") {
		testall();
	}
	else if (FLAGS_action == "test_dump") { // added by dongyu
		test_dump();
	}
	else if (FLAGS_action == "test_dump_model") { // added by dongyu
		test_dump_model();
	}
	else {
		cout << "Please specified a valid action. (train|train_ohem|benchmark|bn_postprocess|box_test|test)" << endl;
	}

	gflags::ShutDownCommandLineFlags();
	return 0;
}
