
#include <caffepro/data/data_entries/rcnn_entry.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/utils/random_helper.h>
#include <algorithm>
#include <omp.h>

namespace caffepro {
	namespace data_model {
		using std::vector;
		using std::string;
		using data_utils::Box;

		rcnn_entry::rcnn_entry(data_provider &provider, const std::string &name)
			: data_entry(provider, name) {
			type_ = "rcnn";
		}

		rcnn_entry::~rcnn_entry() {
			// nothing to do
		}

		void rcnn_entry::init() {
			data_entry::init();

			vector<float> pos_range = config_.get_array<float>("pos_range");
			CHECK_EQ(pos_range.size(), 2);
			CHECK_LE(pos_range[0], pos_range[1]);
			pos_range_.first = pos_range[0];
			pos_range_.second = pos_range[1];

			vector<float> neg_range = config_.get_array<float>("neg_range");
			CHECK_EQ(neg_range.size(), 2);
			CHECK_LE(neg_range[0], neg_range[1]);
			neg_range_.first = neg_range[0];
			neg_range_.second = neg_range[1];

			batch_size_ = provider_.data_provider_config().get<int>("batch_size");
			batch_img_size_ = config_.get<int>("batch_img_size");
			foreground_classes_ = config_.get<int>("foreground_classes");
			max_length_small_object_ = config_.get<int>("max_len_small_object");
			channel_num_ = config_.get<int>("channel_num");
			enable_flip_ = config_.get<bool>("enable_flip", false, true);
			pos_ratio_ = config_.get<float>("pos_ratio");
			neg_ratio_ = config_.get<float>("neg_ratio");
			padding_ratio_ = config_.get<float>("padding_ratio", false, 0.f);
			padding_length_ = config_.get<int>("padding_length", false, 0);
			mean_value_ = config_.get<float>("mean_value", false, 128.0f);

			method_ = config_.get<string>("method");
			auto_init_buffer(0, batch_size_, channel_num_, batch_img_size_, batch_img_size_, false);
			auto_init_buffer(1, batch_size_, 1, 1, 1, true);
			auto_init_buffer(2, batch_size_, 4, 1, 1, true);
		}

		void rcnn_entry::prepare(batch_descriptor &batch) {
			CHECK_EQ(batch.batch_data.size(), batch_size_);

			// assign pos and neg sample
			vector<int> pos_neg_assign(batch_size_); // 0 for pos, 1 for neg

			if (neg_ratio_ < 0 || pos_ratio_ < 0) {
				std::fill(pos_neg_assign.begin(), pos_neg_assign.end(), -1);
			}
			else {
				int neg_number = (int)(batch_size_ * ((float)neg_ratio_ / (pos_ratio_ + neg_ratio_)) + 0.5f);
				memset(&pos_neg_assign[0], 0, sizeof(int)* batch_size_);
				for (int i = 0; i < neg_number; i++) {
					pos_neg_assign[i] = 1;
				}
			}

			srand(std::random_device()());
			std::random_shuffle(pos_neg_assign.begin(), pos_neg_assign.end());
			vector<int> seed(batch_size_);
			for (int i = 0; i < batch_size_; i++) seed[i] = rand();

#pragma omp parallel for schedule(dynamic)
			for (int itemid = 0; itemid < batch_size_; itemid++) {
				srand(seed[itemid]);

				auto &item = batch.batch_data[itemid];
				prepare_one(*item.original_data, *item.processed_data, pos_neg_assign[itemid]);
				CHECK_EQ(item.processed_data->data->channels(), channel_num_) << "Prepared image.channels() != channel_num";
			}

			// fill back rcnn image
			bool fixed_size = (method_ != "resize");
			write_batch_to_blob(batch, prefetch_buffer_[0], fixed_size);

			// fill back rcnn label
			data_type *rcnn_label = prefetch_buffer_[1]->get(0)->write_only_cpu_data();
			for (int i = 0; i < batch_size_; i++) {
				rcnn_label[i] = (data_type)batch.batch_data[i].processed_data->label_id;
			}

			// fill back relative offset between proposal and gt box
			data_type *relv_offset_data = prefetch_buffer_[2]->get(0)->write_only_cpu_data();
			for (int i = 0; i < batch_size_; i++) {
				auto &processed_data = batch.batch_data[i].processed_data;
				Box crop_box = boost::any_cast<Box>(processed_data->additional_data["crop_box"]);
				Box gt_box = boost::any_cast<Box>(processed_data->additional_data["gt_box"]);
				bool flip = boost::any_cast<bool>(processed_data->additional_data["flip"]);

				data_type *data = relv_offset_data + i * 4;
				if (crop_box.valid() && gt_box.valid()) {
					data_type diff_left = (data_type)(gt_box.left - crop_box.left) / crop_box.width();
					data_type diff_top = (data_type)(gt_box.top - crop_box.top) / crop_box.height();
					data_type diff_right = (data_type)(gt_box.right - crop_box.right) / crop_box.width();
					data_type diff_bottom = (data_type)(gt_box.bottom - crop_box.bottom) / crop_box.height();
					if (flip) {
						std::swap(diff_left, diff_right);
						diff_left = -diff_left;
						diff_right = -diff_right;
					}

					data[0] = diff_left;
					data[1] = diff_top;
					data[2] = diff_right;
					data[3] = diff_bottom;
				}
				else {
					data[0] = data[1] = data[2] = data[3] = (data_type)0.f;
				}
			}
		}

		inline void find_max_iou(const Box &ref_box, const vector<Box> &bboxes, double &max_iou, int &max_index) {
			max_index = -1;
			max_iou = 0;
			for (int i = 0; i < (int)bboxes.size(); i++) {
				double iou = ref_box.IoU(bboxes[i]);
				if (iou > max_iou) {
					max_iou = iou;
					max_index = i;
				}
			}
		}

		inline bool in_range(float value, const rcnn_entry::range_f range) {
			return range.first <= value && value <= range.second;
		}

		void rcnn_entry::prepare_one(const data_container &original_image, data_container &processed_image, int pos_neg_assign) {
			cv::Mat im_temp;

			// 1. decode image
			im_temp = cv::imdecode(*original_image.data, CV_LOAD_IMAGE_COLOR);

			// 2. select crop
			int selected_index = -1;

			if (!processed_image.additional_data.count("crop_box")) {
				if (original_image.proposal_boxes.size() > 0) {
					if (pos_neg_assign >= 0) {
						vector<int> candidate_index;
						for (int i = 0; i < (int)original_image.proposal_boxes.size(); i++) {
							double max_gt_iou;
							int max_gt_index;
							find_max_iou(original_image.proposal_boxes[i], original_image.groundtruth_boxes, max_gt_iou, max_gt_index);

							range_f rng = pos_neg_assign == 0 ? pos_range_ : neg_range_;
							if (in_range((float)max_gt_iou, rng)) {
								candidate_index.push_back(i);
							}
						}

						if (candidate_index.size() > 0) {
							selected_index = candidate_index[rand() % candidate_index.size()];
						}
						else {
							// no candidate, random select one
							selected_index = rand() % original_image.proposal_boxes.size();
						}
					}
					else {
						// no need to select pos or neg box
						selected_index = rand() % original_image.proposal_boxes.size();
					}
				}
				else {
					LOG(ERROR) << "WARNING: No proposal for the image " << original_image.data_name << ", use full image instead";
				}
			}

			Box selected_box(0, 0, original_image.width - 1, original_image.height - 1, -1);
			if (selected_index >= 0) selected_box = original_image.proposal_boxes[selected_index];
			if (processed_image.additional_data.count("crop_box")) {
				selected_box = boost::any_cast<Box>(processed_image.additional_data["crop_box"]);
			}

			if (selected_box.valid() && method_ == "resize") { // limits for very large boxes
				const int max_len = 2000;
				float min_ratio = (float)batch_img_size_ / max_len, max_ratio = (float)max_len / batch_img_size_;
				int max_width = (int)(selected_box.height() * max_ratio + 0.5f);
				int max_height = (int)(selected_box.width() / min_ratio + 0.5f);
				
				if (selected_box.width() > max_width) {
					int offset_l = (selected_box.width() - max_width) / 2;
					int offset_r = selected_box.width() - max_width - offset_l;
					selected_box.left += offset_l;
					selected_box.right -= offset_r;
				}

				if (selected_box.height() > max_height) {
					int offset_t = (selected_box.height() - max_height) / 2;
					int offset_b = selected_box.height() - max_height - offset_t;
					selected_box.top += offset_t;
					selected_box.bottom -= offset_b;
				}
			}

			double max_gt_iou;
			int max_gt_index;
			find_max_iou(selected_box, original_image.groundtruth_boxes, max_gt_iou, max_gt_index);

			Box assigned_gt_box = Box::get_invalid_box();
			if (max_gt_index >= 0) {
				assigned_gt_box = original_image.groundtruth_boxes[max_gt_index];
			}

			// 3. get label
			if (in_range((float)max_gt_iou, pos_range_) && assigned_gt_box.label_id >= 0) {
				processed_image.label_id = assigned_gt_box.label_id;
				CHECK_LT(processed_image.label_id, foreground_classes_);
			}
			else {
				processed_image.label_id = foreground_classes_; // background class id equals to number of foreground classes
			}

			// 4. crop image
			cv::Mat crop_patch;
			if (selected_box.valid()) {
				int width_pad = (int)(selected_box.width() * padding_ratio_ + 0.5f);
				int height_pad = (int)(selected_box.height() * padding_ratio_ + 0.5f);

				if (padding_length_ > 0) {
					CHECK_LT(padding_length_ * 2, batch_img_size_);

					if (method_ == "warp") {
						// target_size / (patch_width + 2 * x) = padding_length / x
						// solve x as the new padding length

						width_pad += (selected_box.width() + 2 * width_pad) * padding_length_ / (batch_img_size_ - padding_length_ * 2);
						height_pad += (selected_box.height() + 2 * height_pad) * padding_length_ / (batch_img_size_ - padding_length_ * 2);
					}
					else if (method_ == "resize") {
						int cur_scale = std::min(selected_box.width() + 2 * width_pad, selected_box.height() + 2 * height_pad);
						float resize_scale = (float)(batch_img_size_ - 2 * padding_length_) / cur_scale;
						cv::Size target_size((int)(resize_scale * (selected_box.width() + 2 * width_pad) + 0.5f), (int)(resize_scale * (selected_box.height() + 2 * height_pad) + 0.5f));

						width_pad += (selected_box.width() + 2 * width_pad) * padding_length_ / target_size.width;
						height_pad += (selected_box.height() + 2 * height_pad) * padding_length_ / target_size.height;
					}
				}

				cv::Rect rc(selected_box.left, selected_box.top, selected_box.width(), selected_box.height());
				cv::copyMakeBorder(im_temp(rc), crop_patch, height_pad, height_pad, width_pad, width_pad, cv::BORDER_CONSTANT, cv::Scalar(mean_value_, mean_value_, mean_value_));
			}
			else {
				// invalid image
				crop_patch = cv::Mat(3, 3, im_temp.type());
				crop_patch = cv::Scalar(mean_value_, mean_value_, mean_value_);
			}

			// 5. resize large image patch to smaller one
			int cur_scale = std::min(crop_patch.rows, crop_patch.cols);
			if (cur_scale > max_length_small_object_) {
				float resize_scale = (float)max_length_small_object_ / cur_scale;
				cv::Size new_size((int)(resize_scale * crop_patch.cols + 0.5f), (int)(resize_scale * crop_patch.rows + 0.5f));
				cv::resize(crop_patch, crop_patch, new_size);
			}
			crop_patch.convertTo(crop_patch, CV_32F);

			// 6. minus mean
			crop_patch -= cv::Scalar(mean_value_, mean_value_, mean_value_);

			// 7. flip
			bool flip = false;
			if (!processed_image.additional_data.count("flip")) {
				if (enable_flip_ && rand() % 2) {
					flip = true;
				}
			}
			else {
				flip = boost::any_cast<bool>(processed_image.additional_data["flip"]);
			}

			if (flip) {
				cv::flip(crop_patch, crop_patch, 1);
			}

			// 8. warp or resize image
			processed_image.data.reset(new cv::Mat());

			if (method_ == "warp") {
				cv::Size target_size(batch_img_size_, batch_img_size_);
				cv::resize(crop_patch, *processed_image.data, target_size);
			}
			else if (method_ == "resize") {
				cur_scale = std::min(crop_patch.rows, crop_patch.cols);
				float resize_scale = (float)batch_img_size_ / cur_scale;
				cv::Size target_size((int)(resize_scale * crop_patch.cols + 0.5f), (int)(resize_scale * crop_patch.rows + 0.5f));
				cv::resize(crop_patch, *processed_image.data, target_size);
			}
			else {
				LOG(FATAL) << "Unknown rcnn method: " << method_;
			}

			// 9. save info
			processed_image.additional_data["gt_box"] = assigned_gt_box;
			processed_image.additional_data["crop_box"] = selected_box;
			processed_image.additional_data["flip"] = flip;
		}
	}
}