
#pragma once

#include <caffepro/utils/string_uitls.h>
#include <caffepro/utils/filesystem.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/object_model/node_blob.h>
#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/proto/caffe.pb.h>

#include <caffepro/proto/analyzer_proto.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <caffepro/utils/analyzer_tools.h>

using caffepro::caffepro_layer;
using caffepro::data_model::data_provider;
#define MAX_PROTOFILE_SIZE 5368709120

namespace analyzer {
	
	class DumpInfo {

	private:
		analyzer_proto::Info info;

	public:

		enum save_method : unsigned int {
			SAVE_ALL	= 0U,
			SAVE_GRAD	= 1U,
			SAVE_WEIGHT = 2U,
			NOTSAVE		= 3U
		};

		// DeepTracker-9: save img data to mongodb
		void testRecord(analyzer_proto::Images &imgs, boost::shared_ptr<analyzer_tools::Analyzer> analyzer_tools_instance_) {
			analyzer_tools_instance_->deal_img_info(imgs, 50000);
		}

		void testRecord(analyzer_proto::Images &imgs, int iteration_, std::string foldname = "test_records", std::string filename = "") {

			// filename define
			// AAAAAAAA_BBB.info iteration_rank
			if (filename.empty()) {
				filename = caffepro::fill_zero(iteration_, 8);
			}
			
			if (!caffepro::filesystem::exist(foldname.c_str()))
				caffepro::filesystem::create_directory(foldname.c_str());

			std::string name = foldname + "/" + filename + ".info";
			std::ofstream fp(name.c_str(), std::ios::binary);
			imgs.SerializeToOstream(&fp);
			// add imgs to mongo db directly !!!!!!!!
			fp.close();
		}

		
		// dump all info (grad, weight)
		// SAVE_ALL(default), SAVE_GRAD, SAVE_WEIGHT, NOTSAVE
		void record(
			std::vector<boost::shared_ptr<caffepro_layer>> &layers_,
			boost::shared_ptr<data_provider> &data_provider_,
			int iteration_, int rank_, save_method save_method_ = SAVE_ALL) {

			// DeepTracker-7: save para to mongodb, first convert data type to .info class defined by protobuf

			// filename format: AAAAAAAA_BBB.info iteration_rank --- rank means node number, 0 for main node, others for slave node
			std::string filename = "";
			filename = caffepro::fill_zero(iteration_, 8) + "_" + caffepro::fill_zero(rank_, 3);

			// basic info
			info.set_filename(filename.c_str());
			info.set_iteration(iteration_);
			info.set_worker_id(rank_);

			// save training image info
			if (save_method_ == SAVE_ALL) {
				auto &img_info_ = data_provider_->img_info();
				img_info_->set_iteration(iteration_);
				info.mutable_images()->CopyFrom(*img_info_);
				img_info_->Clear();
			}
			
			// layers info
			for (auto layer_ : layers_) {
				for (auto weight_info_ : layer_->weights()) {

					auto weight = weight_info_->get(0);
					// add one layer to info
					auto nl = info.add_layers();

					nl->set_count(weight->count());
					nl->set_name(layer_->layer_param().name());
					nl->set_num(weight->num());
					nl->set_channels(weight->channels());
					nl->set_height(weight->height());
					nl->set_width(weight->width());
					nl->set_type(layer_->layer_param().type());

					if (save_method_ != NOTSAVE) {
						for (int i = 0; i < nl->count(); i++) {
							if (save_method_ == SAVE_GRAD || save_method_ == SAVE_ALL) {
								nl->add_grad(weight->cpu_diff()[i]);		// gradient info
							}
							if (save_method_ == SAVE_WEIGHT || save_method_ == SAVE_ALL) {
								nl->add_weight(weight->cpu_data()[i]);		// weight info
							}
						}
					}
				}
			}
		}

		// dump to file
		void save_to_file(std::string foldname) {			
			if (!caffepro::filesystem::exist(foldname.c_str()))
				caffepro::filesystem::create_directory(foldname.c_str());

			std::string filename = foldname + "/" + info.filename() + ".info";
			std::ofstream fp(filename.c_str(), std::ios::binary);
			info.SerializeToOstream(&fp);
			fp.close();
		}

		// dump to db
		void save_to_db(boost::shared_ptr<analyzer_tools::Analyzer> analyzer_tools_instance_) {
			// DeepTracker-7: save para to mongodb
			analyzer_tools_instance_->deal_para_info(info);
		}

	};

}