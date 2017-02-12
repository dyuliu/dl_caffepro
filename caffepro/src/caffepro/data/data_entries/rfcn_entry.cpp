
#include <caffepro/data/data_entries/rfcn_entry.h>
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
		using data_utils::BoxF;

		rfcn_entry::rfcn_entry(data_provider &provider, const std::string &name)
			: data_entry(provider, name) {
			type_ = "rfcn";
		}

		rfcn_entry::~rfcn_entry() {
			// nothing to do
		}

		void rfcn_entry::init() {
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
			batch_img_scale_ = config_.get<int>("batch_img_scale");
			batch_img_min_scale_ = config_.get<int>("batch_img_min_scale");
			foreground_classes_ = config_.get<int>("foreground_classes");
			max_scale_small_object_ = config_.get<int>("max_scale_small_object");
			channel_num_ = config_.get<int>("channel_num");
			enable_flip_ = config_.get<bool>("enable_flip", false, true);
			pos_ratio_ = config_.get<float>("pos_ratio");
			neg_ratio_ = config_.get<float>("neg_ratio");
			padding_ratio_ = config_.get<float>("padding_ratio", false, 0.f);
			padding_length_ = config_.get<float>("padding_length", false, 0.f);
			src_padding_length_ = config_.get<float>("src_padding_length", false, 0.f);
			mean_value_ = config_.get<float>("mean_value", false, 128.0f);

			feature_map_padding_ratio_ = config_.get<float>("feature_map_padding_ratio", false, 0.f);
			feature_map_start_ = config_.get<float>("feature_map_start");
			feature_map_scale_ = config_.get<float>("feature_map_scale");

			auto_init_buffer(0, batch_size_, channel_num_, batch_img_scale_, batch_img_scale_, false);	// resized image
			auto_init_buffer(1, batch_size_, 1, 1, 1, true);	// class label (the last class is background)
			auto_init_buffer(2, batch_size_, 4, 1, 1, true);	// feature map crop box (with scale and shift)
			auto_init_buffer(3, batch_size_, 4, 1, 1, true);	// relative diff from proposal box to ground-truth box
			auto_init_buffer(4, batch_size_, 4, 1, 1, true);	// feature map crop box (without scale and shift)
		}

		static bool compare_aspect_ratio(const batch_descriptor::batch_datum &a, const batch_descriptor::batch_datum &b) {
			float ratio_a = boost::any_cast<float>(a.processed_data->additional_data["src_ratio"]);
			float ratio_b = boost::any_cast<float>(b.processed_data->additional_data["src_ratio"]);
			return ratio_a < ratio_b;
		}

		inline void box_shift(BoxF &box, float shift_x, float shift_y) {
			box.left += shift_x;
			box.right += shift_x;
			box.top += shift_y;
			box.bottom += shift_y;
		}

		inline void box_scale(BoxF &box, float scale) {
			box.left *= scale;
			box.right *= scale;
			box.top *= scale;
			box.bottom *= scale;
		}

		inline void box_scale(BoxF &box, float scale_x, float scale_y) {
			box.left *= scale_x;
			box.right *= scale_x;
			box.top *= scale_y;
			box.bottom *= scale_y;
		}

		void rfcn_entry::prepare(batch_descriptor &batch) {
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

			// generate random seed
			srand(std::random_device()());
			std::random_shuffle(pos_neg_assign.begin(), pos_neg_assign.end());
			vector<int> seed(batch_size_);
			for (int i = 0; i < batch_size_; i++) seed[i] = rand();

			// select prop box
#pragma omp parallel for schedule(dynamic)
			for (int itemid = 0; itemid < batch_size_; itemid++) {
				srand(seed[itemid]);
				auto &item = batch.batch_data[itemid];
				select_proposal(*item.original_data, *item.processed_data, pos_neg_assign[itemid]);
			}

			// clustering proposals based on aspect ratio
			std::sort(batch.batch_data.begin(), batch.batch_data.end(), compare_aspect_ratio);
			vector<int> target_width, target_height;
			cluster_proposal(batch, target_width, target_height);
			CHECK_EQ(target_width.size(), batch_size_);
			CHECK_EQ(target_height.size(), batch_size_);

#pragma omp parallel for schedule(dynamic)
			for (int itemid = 0; itemid < batch_size_; itemid++) {
				auto &item = batch.batch_data[itemid];
				prepare_one(*item.original_data, *item.processed_data, target_width[itemid], target_height[itemid]);
				CHECK_EQ(item.processed_data->data->channels(), channel_num_) << "Prepared image.channels() != channel_num";
			}

			// fill back rcnn image
			write_batch_to_blob(batch, prefetch_buffer_[0], false);

			// fill back rcnn label
			data_type *rcnn_label = prefetch_buffer_[1]->get(0)->write_only_cpu_data();
			for (int i = 0; i < batch_size_; i++) {
				rcnn_label[i] = (data_type)batch.batch_data[i].processed_data->label_id;
			}

			// fill back content box (ROI box)
			data_type *content_box = prefetch_buffer_[2]->get(0)->write_only_cpu_data();
			data_type *content_box_without_scaleshift = prefetch_buffer_[4]->get(0)->write_only_cpu_data();
			for (int i = 0; i < batch_size_; i++) {
				BoxF box = boost::any_cast<BoxF>(batch.batch_data[i].processed_data->additional_data["content_box"]);
				box.fill(content_box_without_scaleshift + i * 4);
				box_scale(box, feature_map_scale_);
				box_shift(box, feature_map_start_, feature_map_start_);
				box.fill(content_box + i * 4);
			}

			// fill back relative offset between proposal and gt box
			data_type *relv_offset_data = prefetch_buffer_[3]->get(0)->write_only_cpu_data();
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

		inline bool in_range(float value, const rfcn_entry::range_f range) {
			return range.first <= value && value <= range.second;
		}

		void rfcn_entry::select_proposal(const data_container &original_image, data_container &processed_image, int pos_neg_assign) {
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
			if (processed_image.additional_data.count("crop_box")) {
				selected_box = boost::any_cast<Box>(processed_image.additional_data["crop_box"]);
			}
			else if (selected_index >= 0) {
				selected_box = original_image.proposal_boxes[selected_index];
			}

			if (!selected_box.valid()) {
				selected_box = Box(0, 0, original_image.width - 1, original_image.height - 1, -1);
				LOG(ERROR) << "WARNING: Invalid proposal. Use full image instead";
			}

			// limits for the aspect ratio
			const float max_ratio = 10, min_ratio = 1.f / max_ratio;
			float r = (float)selected_box.width() / selected_box.height();
			if (r < min_ratio) {
				int h = (int)(selected_box.width() / min_ratio + 0.5f);
				int offset_top = (selected_box.height() - h) / 2;
				int offset_bottom = selected_box.height() - h - offset_top;
				selected_box.top += offset_top;
				selected_box.bottom -= offset_bottom;
			}
			else if (r > max_ratio) {
				int w = (int)(selected_box.height() * max_ratio + 0.5f);
				int offset_left = (selected_box.width() - w) / 2;
				int offset_right = selected_box.width() - w - offset_left;
				selected_box.left += offset_left;
				selected_box.right -= offset_right;
			}
			processed_image.additional_data["crop_box"] = selected_box;
			processed_image.additional_data["src_ratio"] =
				(float)(selected_box.width() * (1 + 2 * padding_ratio_) + 2 * src_padding_length_)
				/ (selected_box.height() * (1 + 2 * padding_ratio_) + 2 * src_padding_length_);

			// get label
			double max_gt_iou;
			int max_gt_index;
			find_max_iou(selected_box, original_image.groundtruth_boxes, max_gt_iou, max_gt_index);

			Box assigned_gt_box = Box::get_invalid_box();
			if (max_gt_index >= 0) {
				assigned_gt_box = original_image.groundtruth_boxes[max_gt_index];
			}

			if (in_range((float)max_gt_iou, pos_range_) && assigned_gt_box.label_id >= 0) {
				processed_image.label_id = assigned_gt_box.label_id;
				CHECK_LT(processed_image.label_id, foreground_classes_);
			}
			else {
				processed_image.label_id = foreground_classes_; // background class id equals to number of foreground classes
			}
			
			processed_image.additional_data["gt_box"] = assigned_gt_box;

			// flip
			if (!processed_image.additional_data.count("flip")) {
				processed_image.additional_data["flip"] = enable_flip_ ? (rand() % 2 > 0) : false;
			}
		}

		inline void get_best_size(float ratio, float padding_length, int target_scale, int min_width, int min_height, int &width, int &height) {
			// (w + 2p) * (h + 2p) = s ^ 2
			// w / h = r

			double a = ratio;
			double b = (ratio + 1) * padding_length * 2;
			double c = padding_length * padding_length * 4 - target_scale * target_scale;

			double h = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
			double w = h * ratio;

			if (w < min_width) {
				w = min_width;
			}

			if (h < min_height) {
				h = min_height;
			}

			width = (int)(w + 2 * padding_length + 0.5);
			height = (int)(h + 2 * padding_length + 0.5);
		}

		void rfcn_entry::cluster_proposal(batch_descriptor &batch, std::vector<int> &target_widths, std::vector<int> &target_heights) {
			target_widths.reserve(batch_size_);
			target_heights.reserve(batch_size_);
			vector<int> splits = split_minibatch_;
			if (splits.empty()) splits.push_back(batch_size_);

			int cur_item_index = 0;
			for (auto iter = splits.begin(); iter != splits.end(); ++iter) {
				int split_size = *iter;

				float min_ratio = boost::any_cast<float>(batch.batch_data[cur_item_index].processed_data->additional_data["src_ratio"]);
				float max_ratio = boost::any_cast<float>(batch.batch_data[cur_item_index + split_size - 1].processed_data->additional_data["src_ratio"]);

				int w1, w2, h1, h2;
				get_best_size(min_ratio, padding_length_, batch_img_scale_, batch_img_min_scale_, batch_img_min_scale_, w1, h1);
				get_best_size(max_ratio, padding_length_, batch_img_scale_, batch_img_min_scale_, batch_img_min_scale_, w2, h2);

				for (int i = 0; i < split_size; i++) {
					target_widths.push_back(std::max(w1, w2));
					target_heights.push_back(std::max(h1, h2));
				}
				
				cur_item_index += split_size;
			}
		}

		void rfcn_entry::prepare_one(const data_container &original_image, data_container &processed_image,
			int target_width, int target_height) {
			cv::Mat im_temp;

			// 1. decode image
			im_temp = cv::imdecode(*original_image.data, CV_LOAD_IMAGE_COLOR);

			// 2. get proposal box
			Box selected_box = boost::any_cast<Box>(processed_image.additional_data["crop_box"]);
			float src_ratio = boost::any_cast<float>(processed_image.additional_data["src_ratio"]);

			// 3. crop image
			
			CHECK(selected_box.valid());

			int valid_width, valid_height;
			get_best_size(src_ratio, padding_length_, batch_img_scale_, 0, 0, valid_width, valid_height);
			float extra_pad = (float)selected_box.width() * (1 + 2 * padding_ratio_) / (valid_width - 2 * padding_length_) * padding_length_;

			int width_pad = (int)(selected_box.width() * padding_ratio_ + src_padding_length_ + extra_pad + 0.5f);
			int height_pad = (int)(selected_box.height() * padding_ratio_ + src_padding_length_ + extra_pad + 0.5f);
			float content_width_pad = (float)selected_box.width() * feature_map_padding_ratio_;
			float content_height_pad = (float)selected_box.height() * feature_map_padding_ratio_;
			float content_width = content_width_pad * 2 + (float)selected_box.width();
			float content_height = content_height_pad * 2 + (float)selected_box.height();

			cv::Mat crop_patch;
			cv::Rect rc(selected_box.left, selected_box.top, selected_box.width(), selected_box.height());
			cv::copyMakeBorder(im_temp(rc), crop_patch, height_pad, height_pad, width_pad, width_pad, cv::BORDER_CONSTANT, cv::Scalar(mean_value_, mean_value_, mean_value_));
			BoxF content_box = BoxF(
				(float)width_pad - content_width_pad,
				(float)height_pad - content_height_pad,
				(float)width_pad - content_width_pad + content_width,
				(float)height_pad - content_height_pad + content_height,
				-1
				);
			processed_image.additional_data["context_box"] = BoxF(
				(float)selected_box.left - width_pad,
				(float)selected_box.top - height_pad,
				(float)selected_box.right + width_pad,
				(float)selected_box.bottom + height_pad,
				-1
				);

			// 4. resize large image patch to smaller one
			if (selected_box.area() > max_scale_small_object_ * max_scale_small_object_) { // use area of proposal box as the metric
				int content_width, content_height;
				get_best_size((float)selected_box.width() / selected_box.height(), 0.f, max_scale_small_object_, 0, 0, content_width, content_height);
				float resize_scale = (float)content_width / selected_box.width();
				cv::Size new_size((int)(resize_scale * crop_patch.cols + 0.5f), (int)(resize_scale * crop_patch.rows + 0.5f));
				cv::resize(crop_patch, crop_patch, new_size);
				box_scale(content_box, resize_scale);
			}
			crop_patch.convertTo(crop_patch, CV_32F);

			// 5. minus mean
			crop_patch -= cv::Scalar(mean_value_, mean_value_, mean_value_);

			// 6. flip
			bool flip = boost::any_cast<bool>(processed_image.additional_data["flip"]);
			if (flip) {
				cv::flip(crop_patch, crop_patch, 1);
				BoxF patch_box(0, 0, (float)crop_patch.cols - 1.f, (float)crop_patch.rows - 1.f, -1);
				content_box.flip(patch_box);
			}

			// 7. resize image
			processed_image.data.reset(new cv::Mat());
			
			
			valid_width = std::min(valid_width, target_width); // prevent overflow; generally valid_width <= target_width
			valid_height = std::min(valid_height, target_height); // prevent overflow; generally valid_height <= target_height
			float resize_scale_x = (float)valid_width / crop_patch.cols;
			float resize_scale_y = (float)valid_height / crop_patch.rows;

			cv::Size valid_size(valid_width, valid_height);
			cv::resize(crop_patch, *processed_image.data, valid_size);
			box_scale(content_box, resize_scale_x, resize_scale_y);

			int pad_left = (target_width - valid_width) / 2;
			int pad_right = target_width - valid_width - pad_left;
			int pad_top = (target_height - valid_height) / 2;
			int pad_bottom = target_height - valid_height - pad_top;
			cv::copyMakeBorder(*processed_image.data, *processed_image.data, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
			box_shift(content_box, (float)pad_left, (float)pad_top);

			// 9. save info
			processed_image.additional_data["content_box"] = content_box;
		}
	}
}