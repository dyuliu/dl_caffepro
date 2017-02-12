
#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/updater/updaters.h>
#include <caffepro/solver/sgd_solver.h>
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <omp.h>
#include <iostream>

#include "gradient_checker.h"

using namespace caffepro;

using std::cout;
using std::endl;
using std::string;
using std::ofstream;

void test_arch() {
	const string proto_file = "./samples/test1.prototxt";

	NetParameter net_param;
	proto_io(net_param).from_text_file(proto_file);

	caffepro_context context;

	caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);

	while (true) {
		net->forward(false);
		context.sync_all_devices();
		net->backward(false);
		context.sync_all_devices();
		net->finished_reshape();
	}

	gradient_checker checker(net->input_blobs(), net->output_blobs());
	checker.check_net(*net);
}

void test_model() {
	const string proto_file = "net_test_256_center_new2.prototxt";
	const string proto_bin = "net_final_best";

	NetParameter net_param;
	proto_io(net_param).from_text_file(proto_file);

	caffepro_context context;

	cudaSetDevice(0);

	caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);
	NetParameter weights;
	proto_io(weights).from_binary_file(proto_bin);
	net->load_weights(weights);

	double avg_error = 0;
	int iter = 0;
	for (; iter < 30; iter++) {
		clock_t start_time = clock();
		net->forward(true);
		context.sync_all_devices();
		cout << "time: " << clock() - start_time << endl;

		const data_type *output_data = net->output_blobs()[0]->get(0)->cpu_data();
		data_type error = output_data[0];
		cout << error << endl;
		avg_error += error;

		//net->backward(true);
		//context.sync_all_devices();
		net->finished_reshape();
	}

	cout << "avg error: " << avg_error / iter << endl;
}

void test_data_provider() {
	caffepro_context context;

	caffepro_config config;
	config.load_config("provider_config.txt");

	data_model::data_provider provider(&context, &config);
	provider.set_data_provider_name("provider_cfg");

	provider.auto_build();
	provider.init();

	provider.forward();
}

void test_updater() {
	const string proto_file = "net_train.prototxt";

	NetParameter net_param;
	proto_io(net_param).from_text_file(proto_file);

	caffepro_context context;
	caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);

	std::vector<caffepro::caffepro_object::output_metric> metrics;
	SolverParameter solverparam;
	solverparam.set_momentum((float)0.9);
	sgd_updater updater(&context, net->weights_info(), solverparam, metrics);

	for (int iter = 1; ; ) {
		data_type sum_error = 0, sum_loss = 0;
		for (int k = 0; k < 20; k++, iter++) {
			if (k == 19)
			cout << iter << ": " << endl;
			clock_t start_time = clock();
			net->forward(false);
			context.sync_all_devices();
			if (k == 19)
			cout << "forward time: " << clock() - start_time << endl;

			start_time = clock();
			net->backward(true);
			context.sync_all_devices();
			if (k == 19)
			cout << "backward time: " << clock() - start_time << endl;

			start_time = clock();
			updater.update(0.1f, 0.0001f);
			context.sync_all_devices();
			if (k == 19)
			cout << "update time: " << clock() - start_time << endl;

			const data_type *output_data = net->output_blobs()[0]->get(0)->cpu_data();
			data_type error = output_data[0];
			data_type loss = output_data[1];
			sum_error += error;
			sum_loss += loss;
		}
		cout << sum_error / 20 << " " << sum_loss / 20 << endl;

		cout << endl;
	}
}

void debug_diff() {
	const string proto_file = "net_debug2.prototxt";
	const string proto_bin = "net_final_best";

	NetParameter net_param;
	proto_io(net_param).from_text_file(proto_file);

	caffepro_context context;
	caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);

	NetParameter weights;
	proto_io(weights).from_binary_file(proto_bin);
	net->load_weights(weights);

	std::vector<caffepro::caffepro_object::output_metric> metrics;
	SolverParameter solverparam;
	solverparam.set_momentum((float)0.9);
	sgd_updater updater(&context, net->weights_info(), solverparam, metrics);

	net->forward(false);
	context.sync_all_devices();

	//const string &save_layer = "fc8";
	//ofstream out_act(".\\log\\caffepro_out_fw_" + save_layer + ".csv");
	//net->get_layer(save_layer)->outputs()[0]->get(0)->save_data(out_act);

	net->backward(false);
	context.sync_all_devices();

	const string &save_layer = "bn_conv3_3";
	ofstream out_act(".\\log\\caffepro_out_bpact_" + save_layer + ".csv");
	net->get_layer(save_layer)->inputs()[0]->get(0)->save_diff(out_act);

	ofstream out_weight(".\\log\\caffepro_out_bpweight_" + save_layer + ".csv");
	net->get_layer(save_layer)->weights()[0]->get(0)->save_diff(out_weight);

	ofstream out_bias(".\\log\\caffepro_out_bpbias_" + save_layer + ".csv");
	net->get_layer(save_layer)->weights()[1]->get(0)->save_diff(out_bias);

	ofstream out_act2(".\\log\\caffepro_out_bpact_dev2_" + save_layer + ".csv");
	net->get_layer(save_layer)->inputs()[0]->get(1)->save_diff(out_act2);

	ofstream out_weight2(".\\log\\caffepro_out_bpweight_dev2_" + save_layer + ".csv");
	net->get_layer(save_layer)->weights()[0]->get(1)->save_diff(out_weight2);

	ofstream out_bias2(".\\log\\caffepro_out_bpbias_dev2_" + save_layer + ".csv");
	net->get_layer(save_layer)->weights()[1]->get(1)->save_diff(out_bias2);
}

void check_update() {
	const string proto_file = "net_debug2.prototxt";
	const string proto_bin = "net_final_bnst.model";

	NetParameter net_param;
	proto_io(net_param).from_text_file(proto_file);

	caffepro_context context;
	caffepro_net *net1 = caffepro_net::create_from_proto(&context, net_param);
	caffepro_net *net2 = caffepro_net::create_from_proto(&context, net_param);

	NetParameter weights;
	proto_io(weights).from_binary_file(proto_bin);
	net1->load_weights(weights);
	net2->load_weights(weights);

	std::vector<caffepro::caffepro_object::output_metric> metrics;
	SolverParameter solverparam;
	solverparam.set_momentum((float)0.9);

	sgd_updater updater1(&context, net1->weights_info(), solverparam, metrics);
	sgd_updater_faster updater2(&context, net2->weights_info(), solverparam, metrics);

	net1->forward(false);
	net2->forward(false);
	context.sync_all_devices();
	cout << net1->output_blobs()[0]->get(0)->cpu_data()[0] << endl;
	cout << net2->output_blobs()[0]->get(0)->cpu_data()[0] << endl;

	net1->backward(true);
	net2->backward(true);
	context.sync_all_devices();

	updater1.update(0.1f, 1.f);
	updater2.update(0.1f, 1.f);

	context.sync_all_devices();
	
	auto &weights1 = net1->weights();
	auto &weights2 = net2->weights();

	data_type global_max_diff = 0;
	for (int i = 0; i < (int)weights1.size(); i++) {
		const data_type *data1 = weights1[i]->get((int)weights1[i]->size() - 1)->cpu_data();
		const data_type *data2 = weights2[i]->get((int)weights2[i]->size() - 1)->cpu_data();
		int count = weights1[i]->get(0)->count();

		data_type max_diff = 0;
		for (int j = 0; j < count; j++) {
			max_diff = std::max(max_diff, std::abs(data1[j] - data2[j]));
		}

		global_max_diff = std::max(global_max_diff, max_diff);
		cout << max_diff << endl;
	}

	cout << endl << global_max_diff << endl;
}

void check_sgd_solver() {
	caffepro_context context;
	string solver_file = "net_solver.prototxt";
	string solver_binary_file = "net_cache_iter_160.solverstate";

	sgd_solver solver(&context, solver_file);
	solver.load(solver_binary_file);

	solver.run();
}


//int main(int argc, char *argv[]) {
//	check_update();
//	return 0;
//}
