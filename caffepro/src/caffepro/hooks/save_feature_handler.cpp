
#include <caffepro/hooks/save_feature_handler.h>
#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/layers/softmax_loss_layer.h>
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/utils/data_utils/binary_io.h>
#include <caffepro/utils/string_uitls.h>
#include <caffepro/context/common_names.h>

namespace caffepro {
	save_feature_handler::save_feature_handler(caffepro_context *context) 
		: hook_handler(context), first_run_(true) {
		// nothing to do
	}

	save_feature_handler::~save_feature_handler() {
		close();
	}

	void save_feature_handler::open(const std::string &layer_name, const std::string &file_name) {
		close();

		stream_.open(file_name, std::ios::binary);
		layer_name_ = layer_name;
		first_run_ = true;
	}

	void save_feature_handler::close() {
		stream_.close();
		layer_name_.clear();
		first_run_ = true;
	}

	hook_status save_feature_handler::notify(hook_trigger trigger, caffepro_object &sender, hook_handler_args &args) {
		caffepro_net *net = dynamic_cast<caffepro_net *>(args.content);
		CHECK(net);
		CHECK(net->has_layer(layer_name_));

		caffepro_layer &layer = *net->get_layer(layer_name_);
		CHECK_GT(layer.inputs().size(), 0);

		node_blob *source = layer.inputs()[0].get();
		
		if (layer.layer_param().type() == "softmax_loss") {
			source = dynamic_cast<softmax_loss_layer &>(layer).avg_prob().get();
		}

		CHECK_GT(source->size(), 0);
		CHECK(source->same_inner_count());

		std::vector<std::string> pic_names;
		std::vector<std::string> cls_names;

	/*	caffepro_object *data_source = context_->get_shared_object(net->get_name(), SHAREDOBJNAME_DATASOURCE);
		CHECK(data_source);
		if (dynamic_cast<data_bigfile_layer *>(data_source)) {
			data_bigfile_layer *data_layer = dynamic_cast<data_bigfile_layer *>(data_source);
			for (auto &pic : data_layer->current_batch().processing_imgs) {
				pic_names.push_back(pic.path);
				cls_names.push_back(std::to_string(pic.label_id + 1));
			}
		}
		else {
			LOG(FATAL) << "Unknown data source";
		}*/

		int feature_dim = source->get(0)->inner_count();
		//CHECK_EQ(source->sum_num(), pic_names.size());

	/*	data_utils::binary_writer writer(stream_);
		if (first_run_) {
			writer.write(feature_dim);
			first_run_ = false;
		}*/

		for (int i = 0; i < source->sum_num(); i++) {
			const data_type *data = source->get_cpu_data_across_dev(i);
			for (int j = 0; j < feature_dim; j++) { printf("%.2f ", data[j]); }
			std::cout << std::endl;
		}

		//for (int i = 0; i < pic_names.size(); i++) {
		//	const data_type *data = source->get_cpu_data_across_dev(i);

		//	writer.write(pic_names[i]);
		//	writer.write(cls_names[i]);
		//	stream_.write((const char *)data, feature_dim * sizeof(data_type));
		//}

		//stream_.flush();

		return hook_status();
	}
}