
#pragma once

#include <caffepro/utils/string_uitls.h>
#include <caffepro/utils/filesystem.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/object_model/node_blob.h>
#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/proto/caffe.pb.h>

#include <caffepro/proto/analyzer.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <caffepro/utils/analyzer_tools.h>

using caffepro::caffepro_layer;
using caffepro::data_model::data_provider;
#define MAX_PROTOFILE_SIZE 5368709120

namespace analyzer {

	class RecordInfo {

	public:

		enum class RECORD_TYPE : unsigned int {
			TRAIN_ERROR		= 0U,
			TRAIN_LOSS		= 1U,
			TEST_ERROR		= 2U,
			TEST_LOSS		= 3U,
			FORWARD_TIME	= 4U,
			BACKWARD_TIME	= 5U,
			UPDATE_TIME		= 6U,
			LEARNING_RATE	= 7U
		};

		RecordInfo() {
			nameOfType = std::map < RECORD_TYPE, std::string > {
				{ RECORD_TYPE::TRAIN_ERROR, "train_error" },
				{ RECORD_TYPE::TRAIN_LOSS, "train_loss" },
				{ RECORD_TYPE::TEST_ERROR, "test_error" },
				{ RECORD_TYPE::TEST_LOSS, "test_loss" },
				{ RECORD_TYPE::FORWARD_TIME, "forward_time" },
				{ RECORD_TYPE::BACKWARD_TIME, "backward_time" },
				{ RECORD_TYPE::UPDATE_TIME, "update_time" },
				{ RECORD_TYPE::LEARNING_RATE, "learning_rate" }
			};
		}

		void set_name(std::string name) {
			recorder.set_name(name);
		}

		void add_record(int iteration, RECORD_TYPE type, float value) {
			analyzer::RecordTuple *tup = recorder.add_tuple();
			tup->set_iteration(iteration);
			tup->set_type(nameOfType[type].c_str());
			tup->set_value(value);
		}

		void add_record(int iteration, std::string type, float value) {
			for (auto name : nameOfType) {
				if (type == name.second) {
					add_record(iteration, name.first, value);
					return;
				}
			}
			// couldn't find the default tpye, input user-define
			analyzer::RecordTuple *tup = recorder.add_tuple();
			tup->set_iteration(iteration);
			tup->set_type(type.c_str());
			tup->set_value(value);
		}

		void save_to_file(std::string filename) {
			filename += ".recorder";
			std::ofstream fp(filename.c_str(), std::ios::binary);
			recorder.SerializeToOstream(&fp);
			fp.close();
		}

		void load_from_file(std::string filename) {

			std::ifstream fp(filename.c_str(), std::ios::binary);

			google::protobuf::io::IstreamInputStream fstr(&fp);
			google::protobuf::io::CodedInputStream code_input(&fstr);
			code_input.SetTotalBytesLimit((int)MAX_PROTOFILE_SIZE, (int)MAX_PROTOFILE_SIZE);

			recorder.ParseFromCodedStream(&code_input);

			fp.close();

		}

	private:
		Recorder recorder;
		std::map<RECORD_TYPE, std::string> nameOfType;

	};
	
	class DumpInfo {

	private:
		Info info;

	public:

		enum save_method : unsigned int {
			SAVE_ALL	= 0U,
			SAVE_GRAD	= 1U,
			SAVE_WEIGHT = 2U,
			NOTSAVE		= 3U
		};

		void testRecord(Images &imgs, int iteration_, std::string foldname = "test_records", std::string filename = "") {

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
			// filename define
			// AAAAAAAA_BBB.info iteration_rank
			std::string filename = "";
			filename = caffepro::fill_zero(iteration_, 8) + "_" + caffepro::fill_zero(rank_, 3);
			// basic info
			info.set_filename(filename.c_str());
			info.set_iteration(iteration_);
			info.set_worker_id(rank_);

			// To do: new Image & import it
			if (save_method_ == SAVE_ALL) {
				auto &img_info_ = data_provider_->img_info();
				//CHECK(img_info_);
				//std::cout << "analyzer.h  " << img_info_->images_size() << std::endl;
				img_info_->set_iteration(iteration_);
				info.mutable_images()->CopyFrom(*img_info_);
				img_info_->Clear();
			}
			
			//img_info_->clear_class_name();
			//img_info_->clear_data_name();

			// layers info
			for (auto layer_ : layers_) {
				for (auto weight_info_ : layer_->weights()) {

					auto weight = weight_info_->get(0);
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
								nl->add_grad(weight->cpu_diff()[i]);
							}
							if (save_method_ == SAVE_WEIGHT || save_method_ == SAVE_ALL) {
								nl->add_weight(weight->cpu_data()[i]);
							}
						}
					}
				}
			}
		}

		// dump to file
		void save_to_file(std::string foldname) {
			// add info to mongo db directly !!!!!!!!
			if (!caffepro::filesystem::exist(foldname.c_str()))
				caffepro::filesystem::create_directory(foldname.c_str());

			std::string filename = foldname + "/" + info.filename() + ".info";
			std::ofstream fp(filename.c_str(), std::ios::binary);
			info.SerializeToOstream(&fp);
			fp.close();
		}

	};

}