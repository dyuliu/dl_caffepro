#include <caffepro/object_model/caffepro_net.h>

#include <caffepro/proto/proto_io.h>

#include "gradient_checker.h"

using namespace caffepro;

//#include <iostream>

using namespace std;
/*
int main(int argc, char *argv[])
{
	//const string proto_file = "./samples/test1.prototxt";
	const string proto_file = "./samples/net_train.prototxt";

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
*/