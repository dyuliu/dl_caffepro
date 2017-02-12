
#include <caffepro/context/caffepro_context.h>
#include <caffepro/layers/vision_layers.h>
#include <caffepro/proto/caffe.pb.h>

#include <gradient_checker.h>

#include <iostream>

using namespace caffepro;
using namespace std;

void test_conv() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("conv1");
	param.set_type("conv");

	auto conv_param = param.mutable_conv_param();
	conv_param->set_num_output(100);
	conv_param->set_kernel_size(3);
	conv_param->set_pad(0);
	conv_param->set_stride(2);
	conv_param->set_bias_term(true);
	conv_param->mutable_weight_filler()->set_type("gaussian");
	conv_param->mutable_weight_filler()->set_mean(0);
	conv_param->mutable_weight_filler()->set_std(0.01f);
	conv_param->mutable_bias_filler()->set_type("constant");
	conv_param->mutable_bias_filler()->set_value(1);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);
	
	caffepro_layer *conv_layer = caffepro_layer::create(&context, param);
	conv_layer->bind(inputs, outputs);
	conv_layer->init();
	conv_layer->resize();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*conv_layer);

	//int k = 0;
	//while (true) {
	//	inputs[0]->get(0)->reshape_4d(128, 256, 28, 28);
	//	k++;
	//	conv_layer->resize();
	//	conv_layer->forward();
	//	context.sync_all_devices();

	//	for (int i = 0; i < 10; i++) {
	//		cout << outputs[0]->get(0)->cpu_data()[i] << endl;
	//	}
	//	cout << outputs[0]->get(0)->mean() << endl;
	//	cout << outputs[0]->get(0)->variance() << endl;

	//	system("pause");
	//}
}

void test_correlation() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("co1");
	param.set_type("correlation");

	auto co_param = param.mutable_correlation_param();
	co_param->set_pad(0);
	co_param->set_stride(2);
	co_param->set_bias_term(true);
	co_param->mutable_bias_filler()->set_type("constant");
	co_param->mutable_bias_filler()->set_value(1);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 9, 5, 3, 3, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *co_layer = caffepro_layer::create(&context, param);
	co_layer->bind(inputs, outputs);
	co_layer->init();
	co_layer->resize();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*co_layer);
}

void test_fc() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("fc1");
	param.set_type("inner_product");

	auto fc_param = param.mutable_inner_product_param();
	fc_param->set_num_output(100);

	fc_param->mutable_weight_filler()->set_type("gaussian");
	fc_param->mutable_weight_filler()->set_mean(0);
	fc_param->mutable_weight_filler()->set_std(1.f);
	fc_param->mutable_bias_filler()->set_type("constant");
	fc_param->mutable_bias_filler()->set_value(1);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *fc_layer = caffepro_layer::create(&context, param);
	fc_layer->bind(inputs, outputs);
	fc_layer->init();
	fc_layer->resize();
	fc_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*fc_layer);
}

void test_pool() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("pool1");
	param.set_type("pool");

	auto pool_param = param.mutable_pool_param();
	pool_param->set_kernel_size(3);
	pool_param->set_pad(1);
	pool_param->set_stride(2);
	pool_param->set_size_floor(false);
	pool_param->set_pool(PoolingParameter_PoolMethod_MAX);
	
	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);
	//int count = inputs[0]->get(0)->count();
	//data_type *data = inputs[0]->get(0)->mutable_cpu_data();
	//for (int i = 0; i < count; i++) data[i] = i + 1;

	caffepro_layer *pool_layer = caffepro_layer::create(&context, param);
	pool_layer->bind(inputs, outputs);
	pool_layer->init();
	pool_layer->resize();
	pool_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*pool_layer);
}

void test_box_pool() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("box_pool1");
	param.set_type("box_pool");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 20, 5, 4, 7, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 20, 4, 1, 1, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);
	inputs[1]->get(0)->fill_data(1.f);

	data_type *data = inputs[1]->get(0)->mutable_cpu_data();
	data[0] = 1.f;
	data[1] = 2.f;
	data[2] = 4.f;
	data[3] = 2.0f;

	caffepro_layer *box_pool_layer = caffepro_layer::create(&context, param);
	box_pool_layer->bind(inputs, outputs);
	box_pool_layer->init();
	box_pool_layer->resize();
	box_pool_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*box_pool_layer);
}

void test_relu() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("relu1");
	param.set_type("relu");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *relu_layer = caffepro_layer::create(&context, param);
	relu_layer->bind(inputs, outputs);
	relu_layer->init();
	relu_layer->resize();
	relu_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*relu_layer);
}

void test_prelu() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("prelu1");
	param.set_type("prelu");

	param.mutable_learnable_leak_relu_param()->mutable_relu_leak_param()->set_type("constant");
	param.mutable_learnable_leak_relu_param()->mutable_relu_leak_param()->set_value(0.3f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *prelu_layer = caffepro_layer::create(&context, param);
	prelu_layer->bind(inputs, outputs);
	prelu_layer->init();
	prelu_layer->resize();
	prelu_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*prelu_layer);
}

void test_step_gate() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("step_gate1");
	param.set_type("step_gate");

	auto step_gate = param.mutable_step_gate_param();
	step_gate->set_init_value(2.5f);
	step_gate->set_step_value(0.f);
	step_gate->set_max_value(3.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs = inputs;
	//outputs.resize(1);
	//outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *step_gate_layer = caffepro_layer::create(&context, param);
	step_gate_layer->bind(inputs, outputs);
	step_gate_layer->init();
	step_gate_layer->resize();
	step_gate_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	step_gate_layer->backward();
	step_gate_layer->forward();
	context.sync_all_devices();
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*step_gate_layer);
}

void test_padding() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("padding1");
	param.set_type("padding");
	param.mutable_padding_param()->set_pad(1);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *padding_layer = caffepro_layer::create(&context, param);
	padding_layer->bind(inputs, outputs);
	padding_layer->init();
	padding_layer->resize();
	padding_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*padding_layer);
}

void test_exp() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("exp1");
	param.set_type("exp");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *exp_layer = caffepro_layer::create(&context, param);
	exp_layer->bind(inputs, outputs);
	exp_layer->init();
	exp_layer->resize();
	exp_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*exp_layer);
}

void test_softmax() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("softmax1");
	param.set_type("softmax");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *softmax_layer = caffepro_layer::create(&context, param);
	softmax_layer->bind(inputs, outputs);
	softmax_layer->init();
	softmax_layer->resize();
	softmax_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;
	gradient_checker checker(inputs, outputs);
	checker.check_layer(*softmax_layer);
}

void test_lrn() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("lrn1");
	param.set_type("lrn");
	param.mutable_lrn_param()->set_local_size(99);
	param.mutable_lrn_param()->set_alpha(3.f);
	param.mutable_lrn_param()->set_beta(0.5f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *lrn_layer = caffepro_layer::create(&context, param);
	lrn_layer->bind(inputs, outputs);
	lrn_layer->init();
	lrn_layer->resize();
	lrn_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;
	gradient_checker checker(inputs, outputs);
	checker.check_layer(*lrn_layer);
}

void test_local_norm() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("local_norm1");
	param.set_type("local_norm");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *local_norm_layer = caffepro_layer::create(&context, param);
	local_norm_layer->bind(inputs, outputs);
	local_norm_layer->init();
	local_norm_layer->resize();
	local_norm_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;
	gradient_checker checker(inputs, outputs);
	checker.check_layer(*local_norm_layer);
}

void test_l2norm() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("l2norm1");
	param.set_type("l2norm");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 6, 5, 4, 7, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *l2norm_layer = caffepro_layer::create(&context, param);
	l2norm_layer->bind(inputs, outputs);
	l2norm_layer->init();
	l2norm_layer->resize();
	l2norm_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;
	gradient_checker checker(inputs, outputs);
	checker.check_layer(*l2norm_layer);
}


void test_softmax_loss() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("softmax_loss1");
	param.set_type("softmax_loss");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 2, 1, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 1, 1, 1, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);
	inputs[1]->get(0)->fill_data(1.f);

	caffepro_layer *softmaxloss_layer = caffepro_layer::create(&context, param);
	softmaxloss_layer->bind(inputs, outputs);
	softmaxloss_layer->init();
	softmaxloss_layer->resize();
	softmaxloss_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->cpu_data()[0] << endl;
	cout << outputs[0]->get(0)->cpu_data()[1] << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*softmaxloss_layer);
}

void test_rcnn_loss() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("rcnn_loss1");
	param.set_type("rcnn_loss");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 1, 1, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 1, 1, 1, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);
	inputs[1]->get(0)->fill_data(1.f);

	caffepro_layer *rcnnloss_layer = caffepro_layer::create(&context, param);
	rcnnloss_layer->bind(inputs, outputs);
	rcnnloss_layer->init();
	rcnnloss_layer->resize();
	rcnnloss_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->cpu_data()[0] << endl;
	cout << outputs[0]->get(0)->cpu_data()[1] << endl;

	gradient_checker checker(inputs, outputs);
	inputs[1]->get(0)->fill_data(1.f);
	checker.check_layer(*rcnnloss_layer);
}

void test_dropout() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("dropout1");
	param.set_type("dropout");
	param.mutable_dropout_param()->set_dropout_ratio(0.5f);
	context.set_phase(caffepro_context::TRAIN);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *dropout_layer = caffepro_layer::create(&context, param);
	dropout_layer->bind(inputs, outputs);
	dropout_layer->init();
	dropout_layer->resize();
	dropout_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*dropout_layer);

	delete dropout_layer;
}

void test_eltwise_sum() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("eltwise_sum1");
	param.set_type("eltwise_sum");
	param.mutable_dropout_param()->set_dropout_ratio(0.5f);
	context.set_phase(caffepro_context::TRAIN);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));

	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_like(&context, *inputs[0]->get(0), 0)
		));

	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);
	inputs[1]->get(0)->fill_data(5.5f);

	caffepro_layer *eltsum_layer = caffepro_layer::create(&context, param);
	eltsum_layer->bind(inputs, outputs);
	eltsum_layer->init();
	eltsum_layer->resize();
	eltsum_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*eltsum_layer);
}

void test_eltwise_max() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("eltwise_max1");
	param.set_type("eltwise_max");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));

	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_like(&context, *inputs[0]->get(0), 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(5.6f);
	inputs[1]->get(0)->fill_data(5.5f);

	caffepro_layer *eltmax_layer = caffepro_layer::create(&context, param);
	eltmax_layer->bind(inputs, outputs);
	eltmax_layer->init();
	eltmax_layer->resize();
	eltmax_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*eltmax_layer);
}

void test_eltwise_amax() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("eltwise_amax1");
	param.set_type("eltwise_amax");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));

	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_like(&context, *inputs[0]->get(0), 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(5.6f);
	inputs[1]->get(0)->fill_data(-5.5f);

	caffepro_layer *eltamax_layer = caffepro_layer::create(&context, param);
	eltamax_layer->bind(inputs, outputs);
	eltamax_layer->init();
	eltamax_layer->resize();
	eltamax_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*eltamax_layer);
}

void test_bn() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("bn1");
	param.set_type("batch_norm");

	auto bn_param = param.mutable_batch_norm_param();
	bn_param->set_record_option(BatchNormalizationParameter_RecordOption_NOT_RECORD);
	bn_param->set_keep_mean(true);

	bn_param->mutable_scale_filler()->set_type("constant");
	bn_param->mutable_scale_filler()->set_value(1.f);
	bn_param->mutable_shift_filler()->set_type("constant");
	bn_param->mutable_shift_filler()->set_value(0.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *bn_layer = caffepro_layer::create(&context, param);
	bn_layer->bind(inputs, outputs);
	bn_layer->init();
	bn_layer->resize();
	bn_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*bn_layer);
}

void test_mn() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("mn1");
	param.set_type("mean_norm");

	auto bn_param = param.mutable_batch_norm_param();
	bn_param->set_record_option(BatchNormalizationParameter_RecordOption_USE_RECORD_NORM);

	bn_param->mutable_shift_filler()->set_type("constant");
	bn_param->mutable_shift_filler()->set_value(0.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *mn_layer = caffepro_layer::create(&context, param);
	mn_layer->bind(inputs, outputs);
	mn_layer->init();
	mn_layer->resize();
	mn_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*mn_layer);
}

void test_run_bn() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("runbn1");
	param.set_type("runavg_batch_norm");

	auto bn_param = param.mutable_batch_norm_param();
	bn_param->set_record_option(BatchNormalizationParameter_RecordOption_NOT_RECORD);
	bn_param->set_sliding_window_eval_coeff(0.9f);

	bn_param->mutable_scale_filler()->set_type("constant");
	bn_param->mutable_scale_filler()->set_value(1.f);
	bn_param->mutable_shift_filler()->set_type("constant");
	bn_param->mutable_shift_filler()->set_value(0.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(3.f);

	caffepro_layer *bn_layer = caffepro_layer::create(&context, param);
	bn_layer->bind(inputs, outputs);
	bn_layer->init();
	bn_layer->resize();
	bn_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;
	cout << bn_layer->weights()[2]->get(0)->mean() << endl;
	cout << bn_layer->weights()[2]->get(0)->variance() << endl;
	cout << bn_layer->weights()[3]->get(0)->mean() << endl;
	cout << bn_layer->weights()[3]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*bn_layer);
}

void test_scalebias() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("scalebias1");
	param.set_type("scalebias");

	auto sc_param = param.mutable_scalebias_param();
	
	sc_param->mutable_weight_filler()->set_type("constant");
	sc_param->mutable_weight_filler()->set_value(1.f);
	sc_param->mutable_bias_filler()->set_type("constant");
	sc_param->mutable_bias_filler()->set_value(0.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *sc_layer = caffepro_layer::create(&context, param);
	sc_layer->bind(inputs, outputs);
	sc_layer->init();
	sc_layer->resize();
	sc_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*sc_layer);
}

void test_l1norm() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("l1norm1");
	param.set_type("l1norm");

	auto l1norm_param = param.mutable_scalebias_param();

	
	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *l1norm_layer = caffepro_layer::create(&context, param);
	l1norm_layer->bind(inputs, outputs);
	l1norm_layer->init();
	l1norm_layer->resize();
	l1norm_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*l1norm_layer);
}

void test_euclidean_loss() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("euclidean_loss1");
	param.set_type("euclidean_loss");

	
	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	outputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 5, 4, 7, 0)
		));
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);

	caffepro_layer *l2_layer = caffepro_layer::create(&context, param);
	l2_layer->bind(inputs, outputs);
	l2_layer->init();
	l2_layer->resize();
	l2_layer->forward();
	context.sync_all_devices();

	cout << inputs[0]->get(0)->mean() << endl;
	cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*l2_layer);
}

void test_concat() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("concat1");
	param.set_type("concat");
	param.mutable_concat_param()->set_concat_dim(1);
	context.set_phase(caffepro_context::TRAIN);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 4, 5, 4, 7, 0)
		));

	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 4, 3, 4, 7, 0)
		));

	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(2.f);
	inputs[1]->get(0)->fill_data(1.f);

	caffepro_layer *concat_layer = caffepro_layer::create(&context, param);
	concat_layer->bind(inputs, outputs);
	concat_layer->init();
	concat_layer->resize();
	concat_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*concat_layer);
}

void test_sigmoid() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("sigmoid1");
	param.set_type("sigmoid");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *sigmoid_layer = caffepro_layer::create(&context, param);
	sigmoid_layer->bind(inputs, outputs);
	sigmoid_layer->init();
	sigmoid_layer->resize();
	sigmoid_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*sigmoid_layer);
}

void test_sym_sigmoid() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("sym_sigmoid1");
	param.set_type("sym_sigmoid");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	outputs = inputs;
	//outputs.resize(1);
	//outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *sym_sigmoid_layer = caffepro_layer::create(&context, param);
	sym_sigmoid_layer->bind(inputs, outputs);
	sym_sigmoid_layer->init();
	sym_sigmoid_layer->resize();
	sym_sigmoid_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*sym_sigmoid_layer);
}

void test_cluster_loss() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("cluster_loss1");
	param.set_type("cluster_loss");
	
	param.mutable_cluster_param()->set_num_centers(6);
	param.mutable_cluster_param()->mutable_weight_filler()->set_type("gaussian");
	param.mutable_cluster_param()->mutable_weight_filler()->set_std(2.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(0.5f);

	caffepro_layer *cluster_loss_layer = caffepro_layer::create(&context, param);
	cluster_loss_layer->bind(inputs, outputs);
	cluster_loss_layer->init();
	cluster_loss_layer->resize();
	cluster_loss_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*cluster_loss_layer);
}

void test_multilabel_sigmoid_loss() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("multilabel_sigmoid_cross_entropy_loss1");
	param.set_type("multilabel_sigmoid_cross_entropy_loss");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 20, 1, 1, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 20, 1, 1, 0)
		));
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(1.f);
	inputs[1]->get(0)->fill_data(1.f);

	caffepro_layer *multilabel_loss_layer = caffepro_layer::create(&context, param);
	multilabel_loss_layer->bind(inputs, outputs);
	multilabel_loss_layer->init();
	multilabel_loss_layer->resize();
	multilabel_loss_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->cpu_data()[0] << endl;
	cout << outputs[0]->get(0)->cpu_data()[1] << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*multilabel_loss_layer);
}

void test_grid_generator() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("grid_generator1");
	param.set_type("grid_generator");
	param.mutable_grid_generator_param()->set_method("translation-similar");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 6, 1, 1, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 10, 9, 8, 11, 0)
		));

	outputs.resize(2);
	outputs[0].reset(new node_blob());
	outputs[1].reset(new node_blob());

	inputs[0]->get(0)->fill_data(0.5f);

	caffepro_layer *grid_generator_layer = caffepro_layer::create(&context, param);
	grid_generator_layer->bind(inputs, outputs);
	grid_generator_layer->init();
	grid_generator_layer->resize();
	grid_generator_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*grid_generator_layer);
}

void test_resize_grid() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("resize_grid1");
	param.set_type("resize_grid");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 12, 6, 7, 5, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 12, 1, 1, 1, 0)
		));

	outputs.resize(2);
	outputs[0].reset(new node_blob());
	outputs[1].reset(new node_blob());

	inputs[0]->get(0)->fill_data(0.5f);
	inputs[1]->get(0)->fill_data(1.0f);

	caffepro_config config;
	config.load_config("F:\\models_for_test\\rfcn\\provider_config_rfcn.txt");

	caffepro_layer *resize_grid_layer = caffepro_layer::create(&context, param);
	resize_grid_layer->config().set_config(&config);
	resize_grid_layer->config().set_default_section_name(param.name());
	resize_grid_layer->bind(inputs, outputs);
	resize_grid_layer->init();
	resize_grid_layer->resize();
	resize_grid_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*resize_grid_layer);
}


void test_sample() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("sample1");
	param.set_type("sample");
	param.mutable_sample_param()->set_concat_output(SampleParameter_ConcatOutputOption_NUM);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(3);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 6, 4, 11, 0)
		));
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 9, 3, 12, 0)
		));
	inputs[2].reset(new node_blob());
	inputs[2]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 9, 3, 12, 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(0.5f);
	inputs[1]->get(0)->fill_data(10.5f);
	inputs[2]->get(0)->fill_data(3.5f);

	caffepro_layer *sample_layer = caffepro_layer::create(&context, param);
	sample_layer->bind(inputs, outputs);
	sample_layer->init();
	sample_layer->resize();
	sample_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);

	checker.check_layer(*sample_layer);
}

void test_householder() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("householder1");
	param.set_type("householder");

	param.mutable_householder_param()->set_source(1);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 3, 4, 2, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *householder_layer = caffepro_layer::create(&context, param);
	householder_layer->bind(inputs, outputs);
	householder_layer->init();
	householder_layer->resize();
	householder_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*householder_layer);
}

void test_instance_sample() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("instance_sample1");
	param.set_type("instance_sample");

	param.mutable_instance_sample_param()->set_num(3);
	param.mutable_instance_sample_param()->set_sample_method(caffepro::InstanceSampleParameter_SampleMethod_RAND);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 7, 3, 4, 2, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *instance_sample_layer = caffepro_layer::create(&context, param);
	instance_sample_layer->bind(inputs, outputs);
	instance_sample_layer->init();
	instance_sample_layer->resize();
	instance_sample_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*instance_sample_layer);
}

void test_matrix_mul() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("matrix_mul1");
	param.set_type("matrix_mul");

	param.mutable_matrix_mul_param()->set_trans_a(false);
	param.mutable_matrix_mul_param()->set_trans_b(true);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 11, 6, 1, 4, 0)
		));
	
	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 13, 2, 3, 4, 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);
	inputs[1]->get(0)->fill_data(-1.f);

	caffepro_layer *matrix_mul_layer = caffepro_layer::create(&context, param);
	matrix_mul_layer->bind(inputs, outputs);
	matrix_mul_layer->init();
	matrix_mul_layer->resize();
	matrix_mul_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*matrix_mul_layer);
}

void test_transpose() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("transpose1");
	param.set_type("transpose");

	param.mutable_transpose_param()->set_lead_dim(3);
	auto dims = param.mutable_transpose_param()->mutable_output_dims();

	*dims->Add() = 24;
	*dims->Add() = 0;

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 11, 6, 1, 4, 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *transpose_layer = caffepro_layer::create(&context, param);
	transpose_layer->bind(inputs, outputs);
	transpose_layer->init();
	transpose_layer->resize();
	transpose_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*transpose_layer);
}

void test_transpose4d() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("transpose1");
	param.set_type("transpose4d");

	

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 11, 6, 7, 4, 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *transpose_layer = caffepro_layer::create(&context, param);
	transpose_layer->bind(inputs, outputs);
	transpose_layer->init();
	transpose_layer->resize();
	transpose_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*transpose_layer);
}

void test_matrix_mul_stack() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("matrix_mul_stack1");
	param.set_type("matrix_mul_stack");

	param.mutable_matrix_mul_stack_param()->set_trans_even(false);
	param.mutable_matrix_mul_stack_param()->set_trans_odd(true);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(5);

	for (int i = 0; i < (int)inputs.size(); i++) {
		inputs[i].reset(new node_blob());
		inputs[i]->add(boost::shared_ptr<device_blob>(
			device_blob::create_4d(&context, 24, 6, 1, 4, 0)
			));
		inputs[i]->get(0)->fill_data(-1.f);
	}

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	caffepro_layer *matrix_mul_stack_layer = caffepro_layer::create(&context, param);
	matrix_mul_stack_layer->bind(inputs, outputs);
	matrix_mul_stack_layer->init();
	matrix_mul_stack_layer->resize();
	matrix_mul_stack_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*matrix_mul_stack_layer);
}

void test_learnable_dropout() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("learnable_dropout1");
	param.set_type("learnable_dropout");

	param.mutable_learnable_dropout_param()->set_init_value(0.f);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 11, 6, 1, 2, 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	context.set_phase(caffepro_context::TEST);

	caffepro_layer *learnable_dropout_layer = caffepro_layer::create(&context, param);
	learnable_dropout_layer->bind(inputs, outputs);
	learnable_dropout_layer->init();
	learnable_dropout_layer->resize();
	learnable_dropout_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*learnable_dropout_layer);
}

void test_dim_innerproduct() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("dim_innerproduct1");
	param.set_type("dim_innerproduct");

	param.mutable_dim_innerproduct_param()->set_dim(0);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(2);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 11, 6, 1, 4, 0)
		));

	inputs[1].reset(new node_blob());
	inputs[1]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 11, 6, 1, 4, 0)
		));

	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);
	inputs[1]->get(0)->fill_data(-1.f);

	caffepro_layer *dim_innerproduct_layer = caffepro_layer::create(&context, param);
	dim_innerproduct_layer->bind(inputs, outputs);
	dim_innerproduct_layer->init();
	dim_innerproduct_layer->resize();
	dim_innerproduct_layer->forward();
	context.sync_all_devices();

	//cout << inputs[0]->get(0)->mean() << endl;
	//cout << inputs[0]->get(0)->variance() << endl;
	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*dim_innerproduct_layer);
}

void test_slice() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("slice1");
	param.set_type("slice");
	
	context.set_phase(caffepro_context::TRAIN);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 32, 4, 7, 0)
		));

	//outputs = inputs;
	outputs.resize(2);
	outputs[0].reset(new node_blob());
	outputs[1].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *slice_layer = caffepro_layer::create(&context, param);
	slice_layer->bind(inputs, outputs);
	slice_layer->init();
	slice_layer->resize();
	slice_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	cout << outputs[1]->get(0)->mean() << endl;
	cout << outputs[1]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*slice_layer);
}


void test_dimshuffle() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("dimshuffle1");
	param.set_type("dimshuffle");

	context.set_phase(caffepro_context::TRAIN);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 32, 4, 7, 0)
		));

	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *dimshufle_layer = caffepro_layer::create(&context, param);
	dimshufle_layer->bind(inputs, outputs);
	dimshufle_layer->init();
	dimshufle_layer->resize();
	dimshufle_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*dimshufle_layer);
}

void test_diag_operation() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("diag_operation1");
	param.set_type("diag_operation");

	param.mutable_diag_operation_param()->set_scale(2);
	param.mutable_diag_operation_param()->set_shift(100);

	context.set_phase(caffepro_context::TRAIN);

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 32, 2, 5, 7, 0)
		));

	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *diag_operation = caffepro_layer::create(&context, param);
	diag_operation->bind(inputs, outputs);
	diag_operation->init();
	diag_operation->resize();
	diag_operation->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*diag_operation);
}

void test_diag4d_operation() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("diag4d_operation1");
	param.set_type("diag4d_operation");

	param.mutable_diag_operation_param()->set_scale(2);
	param.mutable_diag_operation_param()->set_shift(100);

	context.set_phase(caffepro_context::TRAIN);
	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 40, 32, 3, 3, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());
	inputs[0]->get(0)->fill_data(-1.f);

	caffepro_layer *diag4d_operation = caffepro_layer::create(&context, param);
	diag4d_operation->bind(inputs, outputs);
	diag4d_operation->init();
	diag4d_operation->resize();
	diag4d_operation->forward();
	context.sync_all_devices();
	
	gradient_checker checker(inputs, outputs);
	checker.check_layer(*diag4d_operation);
}

void test_birelu() {
	caffepro_context context;

	LayerParameter param;
	param.set_name("birelu1");
	param.set_type("birelu");

	caffepro_layer::layer_io_buffer inputs, outputs;
	inputs.resize(1);
	inputs[0].reset(new node_blob());
	inputs[0]->add(boost::shared_ptr<device_blob>(
		device_blob::create_4d(&context, 2, 5, 4, 7, 0)
		));
	//outputs = inputs;
	outputs.resize(1);
	outputs[0].reset(new node_blob());

	//inputs[0]->get(0)->fill_data(-1.f);
	auto& blob = *inputs[0]->get(0);
	ENTER_DEVICE_CONTEXT(blob.device_id())
		CURAND_CHECK(curandGenerateUniform(
		context.get_device(0)->curand_handle(),
		blob.mutable_gpu_data(),
		blob.count()
		));
	EXIT_DEVICE_CONTEXT;

	caffepro_layer *birelu_layer = caffepro_layer::create(&context, param);
	birelu_layer->bind(inputs, outputs);
	birelu_layer->init();
	birelu_layer->resize();
	birelu_layer->forward();
	context.sync_all_devices();

	cout << outputs[0]->get(0)->mean() << endl;
	cout << outputs[0]->get(0)->variance() << endl;

	gradient_checker checker(inputs, outputs);
	checker.check_layer(*birelu_layer);
}

int main() {
	test_run_bn();
}
