

#include <caffepro_ezentry.h>

#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/data/data_accessors/sequential_proposal_accessor.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/utils/data_utils/binary_io.h>
#include <caffepro/utils/data_utils/box.h>
#include <caffepro/layers/vision_layers.h>
#include <fstream>

using std::string;
using std::ofstream;
using std::vector;
using namespace caffepro;
using namespace caffepro::data_model;
using namespace caffepro::data_utils;

namespace caffepro_tools {

	// shared objects
	static caffepro_context g_context_;
	static caffepro_config g_config_;

	struct box_feature {
		string picture_name;
		vector<Box> boxes;
		vector<std::pair<Box, data_type> > reg_boxes;
		vector<std::pair<Box, data_type> > reg_boxes_full;
		vector<vector<data_type> > cls_feature;

		void clear() {
			picture_name.clear();
			boxes.clear();
			reg_boxes.clear();
			reg_boxes_full.clear();
			cls_feature.clear();
		}
	};

	static void init(const string &net_proto_txt, const string &net_proto_bin, caffepro_net *&test_net) {
		NetParameter net_param;
		proto_io(net_param).from_text_file(net_proto_txt);
		CHECK(!net_param.config_file().empty()) << "Missing config file";

		g_config_.load_config(net_param.config_file());
		boost::shared_ptr<data_provider> provider(new data_provider(&g_context_, &g_config_));

		// init data provider
		provider->set_data_provider_name(net_param.data_provider_name());
		provider->auto_build();
		provider->set_data_accessor(boost::shared_ptr<data_accessor>(new sequential_proposal_accessor(*provider)));
		test_net = caffepro_net::create_from_proto(&g_context_, net_param, provider);

		NetParameter weight_param;
		proto_io(weight_param).from_binary_file(net_proto_bin);
		test_net->load_weights(weight_param);
	}

	static void translate_box_result(box_feature &results, const data_container &pic_info, const rcnn_loss_layer *box_layer, int index) {
		Box crop_box = boost::any_cast<Box>(pic_info.additional_data.find("crop_box")->second);
		crop_box.label_id = (int)(*box_layer->correct()->get_cpu_data_across_dev(index) + 0.5f);

		const data_type *fea = box_layer->avg_prob()->get_cpu_data_across_dev(index);
		int dim = box_layer->avg_prob()->get(0)->inner_count(); // use gpu 0 because same dim for all boxes

		results.boxes.push_back(crop_box);
		results.cls_feature.push_back(vector<data_type>(dim));
		memcpy(&results.cls_feature.back()[0], fea, dim * sizeof(data_type));
	}

	static void translate_box_reg_result(box_feature &results, const data_container &pic_info, box_regression_layer *box_reg_layer, int index) {
		const data_type *reg_offset = box_reg_layer->inputs()[0]->get_cpu_data_across_dev(index);

		// find max conf class id (except for the last one [background])
		auto &fea = results.cls_feature.back();
		data_type max_v = -FLT_MAX;
		int max_index = -1;
		for (int i = 0; i < (int)fea.size() - 1; i++) { // skip the last one (background)
			if (fea[i] > max_v) {
				max_v = fea[i];
				max_index = i;
			}
		}

		reg_offset += 4 * max_index;

		// recover predict box
		Box base_box = boost::any_cast<Box>(pic_info.additional_data.find("crop_box")->second);
		bool flip = boost::any_cast<bool>(pic_info.additional_data.find("flip")->second);

		Box predict_box;
		if (!flip) { // no flip
			predict_box.left = (int)(base_box.left + base_box.width() * reg_offset[0] + 0.5f); // left
			predict_box.top = (int)(base_box.top + base_box.height() * reg_offset[1] + 0.5f); // top
			predict_box.right = (int)(base_box.right + base_box.width() * reg_offset[2] + 0.5f); // right
			predict_box.bottom = (int)(base_box.bottom + base_box.height() * reg_offset[3] + 0.5f); // bottom
		}
		else { // flip
			predict_box.left = (int)(base_box.left - base_box.width() * reg_offset[2] + 0.5f); // left
			predict_box.top = (int)(base_box.top + base_box.height() * reg_offset[1] + 0.5f); // top
			predict_box.right = (int)(base_box.right - base_box.width() * reg_offset[0] + 0.5f); // right
			predict_box.bottom = (int)(base_box.bottom + base_box.height() * reg_offset[3] + 0.5f); // bottom
		}

		Box image_box(0, 0, pic_info.width - 1, pic_info.height - 1, -1);
		predict_box = predict_box.intersect(image_box);

		predict_box.label_id = max_index;
		results.reg_boxes.push_back(std::make_pair(predict_box, max_v));
	}

	static void translate_box_reg_result_full(box_feature &results, const data_container &pic_info,	box_regression_layer *box_reg_layer, int index) {
		const data_type *reg_offsets = box_reg_layer->inputs()[0]->get_cpu_data_across_dev(index);
		auto &fea = results.cls_feature.back();

		Box base_box = boost::any_cast<Box>(pic_info.additional_data.find("crop_box")->second);
		bool flip = boost::any_cast<bool>(pic_info.additional_data.find("flip")->second);
		Box image_box(0, 0, pic_info.width - 1, pic_info.height - 1, -1);

		for (int i = 0; i < (int)fea.size() - 1; i++) {
			const data_type *cur_offset = reg_offsets + i * 4;

			Box predict_box;
			if (!flip) { // no flip
				predict_box.left = (int)(base_box.left + base_box.width() * cur_offset[0] + 0.5f); // left
				predict_box.top = (int)(base_box.top + base_box.height() * cur_offset[1] + 0.5f); // top
				predict_box.right = (int)(base_box.right + base_box.width() * cur_offset[2] + 0.5f); // right
				predict_box.bottom = (int)(base_box.bottom + base_box.height() * cur_offset[3] + 0.5f); // bottom
			}
			else { // flip
				predict_box.left = (int)(base_box.left - base_box.width() * cur_offset[2] + 0.5f); // left
				predict_box.top = (int)(base_box.top + base_box.height() * cur_offset[1] + 0.5f); // top
				predict_box.right = (int)(base_box.right - base_box.width() * cur_offset[0] + 0.5f); // right
				predict_box.bottom = (int)(base_box.bottom + base_box.height() * cur_offset[3] + 0.5f); // bottom
			}

			predict_box = predict_box.intersect(image_box);
			predict_box.label_id = i;
			data_type conf = fea[i];
			results.reg_boxes_full.push_back(std::make_pair(predict_box, conf));
		}
	}

	static void save_box_result(ofstream &stream, const box_feature &results) {
		binary_writer writer(stream);

		writer.write<string>(results.picture_name);
		int nboxes = (int)results.boxes.size();
		writer.write<int>(nboxes);

		for (int i = 0; i < nboxes; i++) {
			// write box
			const Box &box = results.boxes[i];
			writer.write<int>(box.left);
			writer.write<int>(box.top);
			writer.write<int>(box.right);
			writer.write<int>(box.bottom);
			writer.write<int>(box.label_id);

			// no conf here
			
			// write class feature
			int fea_dim = (int)results.cls_feature[i].size();
			writer.write<int>(fea_dim);
			stream.write((const char *)&results.cls_feature[i][0], fea_dim * sizeof(data_type));
		}

		stream.flush();
	}

	static void save_box_reg_result(ofstream &stream, const box_feature &results) {
		binary_writer writer(stream);
	
		writer.write<string>(results.picture_name);
		int nboxes = (int)results.reg_boxes.size();
		writer.write<int>(nboxes);

		for (int i = 0; i < nboxes; i++) {
			// write box
			const Box &box = results.reg_boxes[i].first;
			writer.write<int>(box.left);
			writer.write<int>(box.top);
			writer.write<int>(box.right);
			writer.write<int>(box.bottom);
			writer.write<int>(box.label_id);
			writer.write<data_type>(results.reg_boxes[i].second); // conf
		}

		stream.flush();
	}

	static void save_box_reg_result_full(ofstream &stream, const box_feature &results) {
		binary_writer writer(stream);

		writer.write<string>(results.picture_name);
		int nboxes = (int)results.reg_boxes_full.size();
		writer.write<int>(nboxes);

		for (int i = 0; i < nboxes; i++) {
			// write box
			const Box &box = results.reg_boxes_full[i].first;
			writer.write<int>(box.left);
			writer.write<int>(box.top);
			writer.write<int>(box.right);
			writer.write<int>(box.bottom);
			writer.write<int>(box.label_id);
			writer.write<data_type>(results.reg_boxes_full[i].second); // conf
		}

		stream.flush();
	}

	void box_tester(const string &net_proto_txt, const string &net_proto_bin, const string &save_feature_file_name, bool full_result) {
		caffepro_net *test_net = nullptr;
		init(net_proto_txt, net_proto_bin, test_net);

		// get output entry
		rcnn_loss_layer *box_layer = caffepro_net::get_layer<rcnn_loss_layer>(test_net->loss_layers());
		box_regression_layer *box_reg_layer = caffepro_net::get_layer<box_regression_layer>(test_net->loss_layers());

		if (box_reg_layer) {
			box_reg_layer->inputs()[0]->set_attr(node_blob::NF_NET_OUTPUT);
		}

		// runtime structures
		std::set<string> visited_picture_names;
		box_feature current_picture;
		bool finished = false;

		// outputs
		ofstream stream_box, stream_box_reg, stream_box_reg_full;
		if (box_layer && !save_feature_file_name.empty()) {
			stream_box.open(save_feature_file_name + ".locclsfea", std::ios::binary);
		}
		if (box_reg_layer && !save_feature_file_name.empty()) {
			stream_box_reg.open(save_feature_file_name + ".locfea", std::ios::binary);

			if (full_result) {
				stream_box_reg_full.open(save_feature_file_name + ".full.locfea", std::ios::binary);
			}
		}

		// build pic indexes
		const vector<int> &pic_indexes
			= dynamic_cast<sequential_proposal_accessor *>(test_net->data_provider()->get_data_accessor().get())->data_indexes();
		vector<string> pic_names;
		for (int i = 0; i < (int)pic_indexes.size(); i++) {
			pic_names.push_back(test_net->data_provider()->get_data(pic_indexes[i])->data_name);
		}

		// runtime stat
		int turn = 0, seg_turn = 0;
		double sum_score = 0., sum_seg_score = 0.;
		int current_pic_index = 0;
		device_blob &output_blob = *box_layer->outputs()[0]->get(0);

		while (!finished) {
			turn++;
			seg_turn++;
			
			test_net->forward();
			g_context_.sync_all_devices();

			auto &batch = test_net->data_provider()->current_batch()->batch_data;

			// prepare data
			// since original data may be shuffled within a batch
			// we must arrange the order
			std::multimap<string, int> batch_pic_indexes;
			for (int i = 0; i < (int)batch.size(); i++) {
				const string &pic_name = batch[i].processed_data->data_name;
				if (current_picture.picture_name == pic_name || !visited_picture_names.count(pic_name)) {
					batch_pic_indexes.insert(std::make_pair(pic_name, i));
				}
			}
			vector<int> batch_order;
			for (; current_pic_index < (int)pic_names.size(); current_pic_index++) {
				const string &pic_name = pic_names[current_pic_index];
				if (batch_pic_indexes.count(pic_name) > 0) {
					auto range = batch_pic_indexes.equal_range(pic_name);
					for (auto iter = range.first; iter != range.second; ++iter) {
						batch_order.push_back(iter->second);
					}
				}
				else if (batch_order.size() == batch_pic_indexes.size()) {
					break;
				}
			}
			current_pic_index--;

			if (batch_order.empty()) { // reach tail
				break;
			}

			for (int idx = 0; idx < (int)batch_order.size(); idx++) {
				int i = batch_order[idx];

				const auto &pic_info = *batch[i].processed_data;
				const string &pic_name = pic_info.data_name;

				if (current_picture.picture_name != pic_name) {

					if (!current_picture.picture_name.empty()) {
						// save feature
						if (stream_box.is_open()) {
							save_box_result(stream_box, current_picture);
						}

						if (stream_box_reg.is_open()) {
							save_box_reg_result(stream_box_reg, current_picture);
						}

						if (stream_box_reg_full.is_open()) {
							save_box_reg_result_full(stream_box_reg_full, current_picture);
						}
					}

					// move to next
					if (!visited_picture_names.count(pic_name)) {
						current_picture.clear();
						current_picture.picture_name = pic_name;
						visited_picture_names.insert(pic_name);
					}
					else { // finished working
						finished = true;
						break;
					}
				}

				// record feature
				if (stream_box.is_open()) {
					translate_box_result(current_picture, pic_info, box_layer, i);
				}

				if (stream_box_reg.is_open()) {
					translate_box_reg_result(current_picture, pic_info, box_reg_layer, i);
				}

				if (stream_box_reg_full.is_open()) {
					translate_box_reg_result_full(current_picture, pic_info, box_reg_layer, i);
				}
			}

			data_type err = output_blob.cpu_data()[0];
			sum_score += err;
			sum_seg_score += err;

			if (turn % 10 == 0) {
				LOG(ERROR) << "Processed " << visited_picture_names.size();
				LOG(ERROR) << "Error = " << sum_seg_score / seg_turn;
				sum_seg_score = 0.;
				seg_turn = 0;
			}
		}

		LOG(ERROR) << "Total Error = " << sum_score / turn;
		delete test_net;
	}
}