
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/filesystem.h>
#include <caffepro/utils/data_utils/xml_image.h>
#include <caffepro/utils/string_uitls.h>

#include <fstream>

using std::string;
using std::vector;
using std::ifstream;

namespace caffepro {

	using data_utils::Box;
	using data_utils::raw_picture;

	void data_bigfile_layer::init_classid(std::vector<std::string> &folder_pathes, int num_classes) {
		for (int i = 0; i < num_classes; i++) {
			string labelname = filesystem::get_file_name_without_extension(folder_pathes[i]);

			database_.class_info.label2classID[labelname] = i;
			database_.class_info.classID2label[i] = labelname;
		}
	}

	void data_bigfile_layer::init_picture_database() {
		int batchimgsize = layer_param_.data_bigfile_param().batch_img_size();
		int data_blob_num = layer_param_.data_bigfile_param().channel_num_size();

		// get channel num
		int channel_num_total = 0;
		for (int i = 0; i < data_blob_num; ++i)
			channel_num_total += this->layer_param_.data_bigfile_param().channel_num(i);

		// build folder list
		vector<string> vFileList;
		vFileList = filesystem::get_files(this->layer_param_.data_bigfile_param().source().c_str(), "*.big", false);

		if (vFileList.empty()) {
			LOG(FATAL) << "Big File Folder " << this->layer_param_.data_bigfile_param().source() << " is missing or empty";
		}

		// build classes
		int valid_classes = (int)vFileList.size();

		init_classid(vFileList, valid_classes);

		// load color KL matrics
		if (this->layer_param_.data_bigfile_param().has_color_kl_dir()) {
			data_utils::load_color_kl(database_.class_info, this->layer_param_.data_bigfile_param().color_kl_dir());
			LOG(INFO) << "Loading KL matrics...Done";
		}

		// load pictures and caching data
		// if metadata is defined and cache data is not needed, skip this step
		if (this->layer_param_.data_bigfile_param().cache_data()
			|| !this->layer_param_.data_bigfile_param().has_metadata_file()) {
			LOG(INFO) << "Loading data: " << valid_classes << " classes...";
			size_t image_bytes = 0;
			for (int i = 0; i<valid_classes; i++) {
				LOG(INFO) << "Loading " << database_.class_info.classID2label[i] << " (" << i << ")";
				google::FlushLogFiles(0);
				image_bytes += data_utils::load_bigfile(
					database_.imgs,
					i,
					vFileList[i],
					view_num(data_utils::crop_type(this->layer_param_.data_bigfile_param().crop_type())),
					this->layer_param_.data_bigfile_param().cache_data());
			}
			LOG(INFO) << "Data size: " << image_bytes;
			LOG(INFO) << "Loading data finished";
		}

		// load metadata if needed
		if (this->layer_param_.data_bigfile_param().has_metadata_file()) {
			data_utils::load_metadata(
				this->layer_param_.data_bigfile_param().metadata_file(),
				this->layer_param_.data_bigfile_param().source(),
				view_num(data_utils::crop_type(this->layer_param_.data_bigfile_param().crop_type())),
				database_.class_info.label2classID,
				database_.imgs
				);
		}

		// load proposals if needed
		if (this->layer_param_.data_bigfile_param().has_crop_box_file()) {
			data_utils::load_prop_boxes(
				this->layer_param_.data_bigfile_param().crop_box_file(),
				database_.imgs
				);

			if (data_utils::crop_type(this->layer_param_.data_bigfile_param().crop_type()) == data_utils::CropType_PropWarp) {
				// adjust view num

				int nobox = 0, invalid_box = 0, box_outofrange = 0;
				for (auto iter = database_.imgs.begin(); iter != database_.imgs.end(); ++iter) {
					//CHECK_GT(iter->prop_boxes.size(), 0);
					if (iter->prop_boxes.size() == 0) {
						nobox++;
					}
					else {
						iter->views.resize(iter->prop_boxes.size());

						for (int i = 0; i < (int)iter->prop_boxes.size(); i++) {
							iter->views[i] = i;

							Box box = iter->prop_boxes[i];
							if (!box.valid()) {
								invalid_box++;
							}
							else if (box.left < 0 || box.top < 0 || box.right >= iter->width || box.bottom >= iter->height) {
								box_outofrange++;
								iter->prop_boxes[i] = Box::get_invalid_box();
							}
						}
					}
				}

				LOG(ERROR) << "Pictures without boxes: " << nobox;
				LOG(ERROR) << "Invalid boxes: " << invalid_box;
				LOG(ERROR) << "Boxes out of range: " << box_outofrange;
			}
		}

		// set image index
		current_batch_state_.image_start_idx = 0;
		prefetch_.prefetch_batch_state.image_start_idx = 0;

		// default view
		current_batch_state_.fixed_view = -1;
		prefetch_.prefetch_batch_state.fixed_view = -1;

		// initialize batches
		if (this->layer_param_.data_bigfile_param().random_shuffle())
			std::random_shuffle(database_.imgs.begin(), database_.imgs.end());

		// check if we want to have mean
		if (this->layer_param_.data_bigfile_param().has_mean_file()) {
			LOG(INFO) << "Loading mean file from" << this->layer_param_.data_bigfile_param().mean_file();
			try {
				data_utils::xml_image::load_image(database_.data_mean, this->layer_param_.data_bigfile_param().mean_file());
			}
			catch (std::runtime_error err)
			{
				LOG(FATAL) << "Fail to load mean file";
			}
			CHECK_EQ(database_.data_mean.channels(), channel_num_total);
			
			if (database_.data_mean.rows != batchimgsize || database_.data_mean.cols != batchimgsize)
			{
				LOG(ERROR) << "Warning: database_.data_mean.rows != batchimgsize || database_.data_mean.cols != batchimgsize";
				cv::resize(database_.data_mean, database_.data_mean, cv::Size(batchimgsize, batchimgsize), 0.0, 0.0);
			}

			CHECK_EQ(database_.data_mean.rows, batchimgsize);
			CHECK_EQ(database_.data_mean.cols, batchimgsize);
		}
		else {
			// Simply initialize an all-empty mean.
			LOG(INFO) << "No mean file loaded ";
			database_.data_mean = cv::Mat();
		}

		// if necessary, load multi-label def file
		if (this->layer_param_.data_bigfile_param().has_multilabel_def_file()) {
			LOG(INFO) << "Loading multi-label def file";

			ifstream stream(this->layer_param_.data_bigfile_param().multilabel_def_file());

			string line;
			while (std::getline(stream, line)) {
				line = trim(line);
				if (line.empty()) continue;

				vector<string> sp = split(line, '\t');
				CHECK_GT(sp.size(), 1) << line;

				string elem_name = sp[0];
				vector<int> labels;
				for (int i = 1; i < (int)sp.size(); i++) {
					labels.push_back(string_to_int(sp[i]));
					database_.multilabel_classes = std::max(database_.multilabel_classes, labels.back() + 1);
				}

				if (endwith_ignorecase(elem_name, ".jpeg") || endwith_ignorecase(elem_name, ".jpg")) {
					CHECK_EQ(database_.picname_to_multilabel.count(elem_name), 0) << "Dup pic name in multi-label def file: " << elem_name;
					database_.picname_to_multilabel[elem_name] = labels;
				}
				else {
					CHECK_EQ(database_.clsname_to_multilabel.count(elem_name), 0) << "Dup class name in multi-label def file: " << elem_name;
					database_.clsname_to_multilabel[elem_name] = labels;
				}
			}

			LOG(INFO) << database_.picname_to_multilabel.size() << " picture defs, " << database_.clsname_to_multilabel.size() << " class defs loaded in def file";
			LOG(INFO) << "Different labels: " << database_.multilabel_classes;

			// checking 
			LOG(INFO) << "Checking def file";
			for (vector<raw_picture>::iterator iter = database_.imgs.begin(); iter != database_.imgs.end(); ++iter) {
				string pic_name = iter->path;

				if (!database_.picname_to_multilabel.count(pic_name)) {
					string cls_name = database_.class_info.classID2label[iter->label_id];
					if (!database_.clsname_to_multilabel.count(cls_name)) {
						LOG(FATAL) << "Picture " << pic_name << " in class " << cls_name << " not defined in multi-label def file";
					}
				}
			}
		}
	}

	void data_bigfile_layer::setup_extra_data_processer(int data_extra_num) {
		int batchsize = layer_param_.data_bigfile_param().batch_size();

		prefetch_.extra_data.resize(data_extra_num);

		for (int p = 0; p < layer_param_.data_bigfile_param().additional_data_processer_size(); p++) {
			auto &cur_processer = layer_param_.data_bigfile_param().additional_data_processer(p);

			if (cur_processer.processer_type() == "rcnn_label") {
				CHECK_EQ(cur_processer.binding_output_index_size(), 1);

				int binding_index = cur_processer.binding_output_index(0);
				CHECK_GE(binding_index, 1 + 1);
				CHECK_LT(binding_index, outputs_.size());
				CHECK_LT(binding_index - 1 - 1, data_extra_num);

				prefetch_.extra_data[binding_index - 1 - 1].reset(new node_blob());
				prefetch_.extra_data[binding_index - 1 - 1]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(nullptr, batchsize, 1, 1, 1, context_->get_current_device()->device_id())
					));
			}
			else if (cur_processer.processer_type() == "bounding_box") {
				CHECK_EQ(cur_processer.binding_output_index_size(), 1);
				CHECK_GE(cur_processer.method_size(), 1);
				CHECK(layer_param_.data_bigfile_param().has_metadata_file());

				int binding_index = cur_processer.binding_output_index(0);
				CHECK_GE(binding_index, 1 + 1);
				CHECK_LT(binding_index, outputs_.size());
				CHECK_LT(binding_index - 1 - 1, data_extra_num);

				prefetch_.extra_data[binding_index - 1 - 1].reset(new node_blob());
				prefetch_.extra_data[binding_index - 1 - 1]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(nullptr, batchsize, 8, 1, 1, context_->get_current_device()->device_id())
					));
			}
			else {
				LOG(FATAL) << "Unknown data processer type: " << cur_processer.processer_type();
			}
		}
	}

	int data_bigfile_layer::view_num(data_utils::crop_type croptype) {
		switch (croptype) {
		case data_utils::CropType_Random:
			return 1;
		case data_utils::CropType_10View:
			return 10;
		case data_utils::CropType_Center:
			return 1;
		case data_utils::CropType_18View:
			return 18;
		case data_utils::CropType_MultiCrop:
			CHECK(layer_param_.data_bigfile_param().has_crop_dim1_segs());
			CHECK(layer_param_.data_bigfile_param().has_crop_dim2_segs());
			return layer_param_.data_bigfile_param().crop_dim1_segs()
				* layer_param_.data_bigfile_param().crop_dim2_segs() * 2; // including flip
		case data_utils::CropType_PropWarp:
			return 1;	// when warpping, the view num is determined by the number of prop boxes. here return 1 for init
		case data_utils::CropType_FullView:
			return 2;	// original and the flipped version

		default:
			LOG(FATAL) << "CropImg::Undefined Crop Type!";
			break;
		}

		return -1;
	}

	bool data_bigfile_layer::is_fixed_size(data_utils::crop_type croptype) {
		// currently, only full-view crop is not fixed size
		return croptype != data_utils::CropType_FullView;
	}
}