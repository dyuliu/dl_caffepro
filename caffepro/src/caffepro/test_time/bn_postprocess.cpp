
#include <caffepro/test_time/bn_postprocess.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/utils/filesystem.h>

namespace caffepro {
	using std::vector;
	using std::string;

	bn_postprocess::bn_postprocess(caffepro_context *context, const std::string &net_def_file, const std::string &net_binary_file, const analyzer::Info &info)
		: context_(context) {

		CHECK(context_);
		NetParameter net_def, net_binary;
		proto_io(net_binary).from_binary_file(net_binary_file);
		proto_io(net_def).from_text_file(net_def_file);

		net_.reset(caffepro_net::create_from_proto(context_, net_def));
		net_->load_weights_from_info(net_binary, info);
	}

	bn_postprocess::bn_postprocess(caffepro_context *context, const std::string &net_def_file, const std::string &net_binary_file) 
		: context_(context) {

		CHECK(context_);
		NetParameter net_def, net_binary;
		proto_io(net_def).from_text_file(net_def_file);
		proto_io(net_binary).from_binary_file(net_binary_file);
		init(net_def, net_binary);
	}

	bn_postprocess::bn_postprocess(caffepro_context *context, NetParameter &net_def, NetParameter &net_binary)
		: context_(context) {
		CHECK(context_);

		init(net_def, net_binary);
	}

	void bn_postprocess::init(NetParameter &net_def, NetParameter &net_binary) {
		net_.reset(caffepro_net::create_from_proto(context_, net_def));
		net_->load_weights(net_binary);
	}

	void bn_postprocess::run(int turn) {

		context_->set_phase(caffepro_context::TRAIN);

		vector<caffepro_layer *> bn_layers;

		for (auto iter = net_->layers().begin(); iter != net_->layers().end(); ++iter) {
			string layer_type = (*iter)->layer_param().type();

			if (layer_type == "batch_norm" || layer_type == "mean_norm" || layer_type == "runavg_batch_norm") {
				bn_layers.push_back(iter->get());
			}
		}

		LOG(ERROR) << "There are " << bn_layers.size() << " BN layers to process";

		// drop the first forward (reshape all layers)
		net_->forward(true);
		net_->finished_reshape();

		for (int nl = 0; nl < (int)bn_layers.size(); nl++) {

			if (context_->get_signal(caffepro_context::SIGNAL_STOP_ALL)) {
				break;
			}

			//std::cout << bn_layers[nl]->layer_param().name();

			//LOG(ERROR) << "Processing " << nl+1 << "/" << bn_layers.size();
			caffepro_layer *cur_bn_layer = bn_layers[nl];

			// modify record option
			string layer_type = cur_bn_layer->layer_param().type();
			string layer_name = cur_bn_layer->layer_param().name();

			LayerParameter &param = const_cast<LayerParameter &>(cur_bn_layer->layer_param());

			if (layer_type == "batch_norm" || layer_type == "mean_norm" || layer_type == "runavg_batch_norm") {
				param.mutable_batch_norm_param()->set_record_option(BatchNormalizationParameter_RecordOption_RECORD);
			}

			// forward
			for (int it = 0; it < turn; it++) {
				int forward_index = 0;
				net_->forward_data_provider();
				net_->forward_until(layer_name, true);
				context_->sync_all_devices();
			}

			if (layer_type == "batch_norm" || layer_type == "runavg_batch_norm") {
				net_->get_layer(layer_name)->weights()[2]->broadcast_data_via_gpu(0);
				net_->get_layer(layer_name)->weights()[3]->broadcast_data_via_gpu(0);
			}
			else if (layer_type == "mean_norm") {
				net_->get_layer(layer_name)->weights()[1]->broadcast_data_via_gpu(0);
			}

			// modify record option
			if (layer_type == "batch_norm" || layer_type == "mean_norm" || layer_type == "runavg_batch_norm") {
				param.mutable_batch_norm_param()->set_record_option(BatchNormalizationParameter_RecordOption_USE_RECORD_NORM);
			}
		}
	}

	void bn_postprocess::save(NetParameter &net_binary) {
		net_->save_proto(net_binary);
	}

	void bn_postprocess::save(const std::string &filename) {
		
		std::string foldname = "models";
		if (!caffepro::filesystem::exist(foldname.c_str()))
			caffepro::filesystem::create_directory(foldname.c_str());

		std::string name = foldname + "/" + filename + ".model";

		if (!context_->get_signal(caffepro_context::SIGNAL_STOP_ALL)) {
			NetParameter net_binary;
			net_->save_proto(net_binary);
			std::ofstream fp(name.c_str(), std::ios::binary);
			net_binary.SerializeToOstream(&fp);
			fp.close();
		}
	}
}