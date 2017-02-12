
#include <caffepro/data/data_entries/image_entry.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/utils/data_utils/xml_image.h>
#include <caffepro/utils/data_utils/box_trans.h>
#include <caffepro/utils/data_utils/color_kl.h>
#include <caffepro/utils/random_helper.h>
#include <numeric>

namespace caffepro {
	namespace data_model {
		using std::string;
		using std::vector;
		using data_utils::crop_type;
		using data_utils::Box;

		image_entry::image_entry(data_provider &provider, const std::string &name) 
			: data_entry(provider, name) {
			type_ = "image";
		}

		image_entry::~image_entry() {
			// nothing to do
		}

		void image_entry::init() {
			data_entry::init();

			// load mean
			string mean_file = config_.get<string>("mean_file", false, "");
			if (!mean_file.empty()) {
				LOG(INFO) << "Loading mean file from " << mean_file;
				try {
					data_utils::xml_image::load_image(data_mean_, mean_file);
				}
				catch (std::runtime_error err) {
					LOG(FATAL) << "Fail to load mean file";
				}
			}

			const int batchsize = provider_.data_provider_config().get<int>("batch_size");
			const int batch_img_size = config_.get<int>("batch_img_size");
			const int channel_num_total = config_.get<int>("channel_num");

			auto_init_buffer(0, batchsize, channel_num_total, batch_img_size, batch_img_size, false);
		}

		void image_entry::prepare(batch_descriptor &batch) {
			// parameters
			caffepro_config_reader &provider_config = provider_.data_provider_config();
			caffepro_config_reader &entry_config = config_;

			const int batchsize = provider_config.get<int>("batch_size");
			const int channel_num_total = entry_config.get<int>("channel_num");
			crop_type croptype = crop_type(entry_config.get<int>("crop_type"));
			CHECK_EQ(batch.batch_data.size(), batchsize);

			// move buffer heads to cpu
			for (int nd = 0; nd < (int)prefetch_buffer_[0]->size(); nd++) {
				prefetch_buffer_[0]->get(nd)->write_only_cpu_data();
			}

#pragma omp parallel for schedule(dynamic)
			for (int itemid = 0; itemid < batchsize; itemid++) {
				auto &item = batch.batch_data[itemid];
				prepare_one_image(*item.original_data, *item.processed_data);
				CHECK_EQ(item.processed_data->data->channels(), channel_num_total) << "Prepared image.channels() != sum(channel_num)";
			}

			// write back images
			// currently, only full-view crop is not fixed size
			bool fixed_size = (croptype != data_utils::CropType_FullView);
			write_batch_to_blob(batch, prefetch_buffer_[0], fixed_size);
		}

		void image_entry::prepare_one_image(const data_container &original_image, data_container &processed_image) {
			caffepro_config_reader &provider_config = provider_.data_provider_config();
			caffepro_config_reader &entry_config = config_;

			cv::Mat im_temp, im_cropped;

			// 1. decode image
			im_temp = cv::imdecode(*original_image.data, CV_LOAD_IMAGE_COLOR);

			if (!provider_config.get<string>("metadata_file", false, "").empty()) {
				CHECK_EQ(original_image.width, im_temp.cols);
				CHECK_EQ(original_image.height, im_temp.rows);
			}

			// 2. crop image
			int iViewIdx = -1;
			if (processed_image.additional_data.count("view_id")) {
				iViewIdx = boost::any_cast<int>(processed_image.additional_data["view_id"]);
			}
			bool bFlipLR = false;
			crop_type croptype = crop_type(entry_config.get<int>("crop_type"));
			switch (croptype) {
			case data_utils::CropType_Random:
				bFlipLR = random_helper::uniform_int() % 2 != 0;
				break;
			case data_utils::CropType_10View:
				if (iViewIdx < 0) iViewIdx = random_helper::uniform_int(0, 9);
				bFlipLR = iViewIdx >= 5 ? true : false;
				break;
			case data_utils::CropType_18View: {
				if (iViewIdx < 0) iViewIdx = random_helper::uniform_int(0, 17);
				const bool fl[] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 };
				bFlipLR = fl[iViewIdx];
				break;
			}
			case data_utils::CropType_MultiCrop: {
				int segs = entry_config.get<int>("crop_dim1_segs") * entry_config.get<int>("crop_dim2_segs");
				bFlipLR = (iViewIdx / segs % 2 == 1);
				break;
			}
			case data_utils::CropType_PropWarp:
				if (provider_.context()->get_phase() == caffepro_context::TRAIN) {
					bFlipLR = random_helper::uniform_int() % 2 != 0;
				}
				break;
			case data_utils::CropType_FullView:
				bFlipLR = iViewIdx % 2 != 0;
				break;
			}

			data_utils::crop_position crop_position = data_utils::crop_position(iViewIdx + data_utils::CropPosition_Begin + 1);

			vector<Box> bboxes = original_image.groundtruth_boxes;
			if (bFlipLR) { // box positions should also be flipped
				Box image_box(0, 0, original_image.width - 1, original_image.height - 1, -1);

				for (int i = 0; i < (int)bboxes.size(); i++) {
					bboxes[i] = bboxes[i].flip(image_box);
				}
			}

			int batch_img_size = entry_config.get<int>("batch_img_size");

			cv::Rect crop_rect;
			cv::Rect crop_rect2; // for RCNN, since pad may applied, we should backup the original crop rect
			float scale_selected = 0;

			cv::Rect max_IoU_crop;
			float max_IoU_value = 0;

			int max_crop_turn = 50;
			while (max_crop_turn--) {
				if (croptype == data_utils::CropType_PropWarp) {
					scale_selected = 1;
					if (original_image.proposal_boxes.size() > 0 && original_image.proposal_boxes[iViewIdx % original_image.proposal_boxes.size()].valid()) {
						Box prop_box = original_image.proposal_boxes[iViewIdx % original_image.proposal_boxes.size()];
						crop_rect2 = cvRect(prop_box.left, prop_box.top, prop_box.width(), prop_box.height());	// backup original box

						if (entry_config.exist("rcnn_pad")) {
							float pad_ratio = entry_config.get<float>("rcnn_pad");
							int pad_width = (int)((prop_box.width() * pad_ratio) / 2), pad_height = (int)((prop_box.height() * pad_ratio) / 2);
							prop_box.left -= pad_width;
							prop_box.top -= pad_height;
							prop_box.right += pad_width;
							prop_box.bottom += pad_height;
							prop_box = prop_box.intersect(Box(0, 0, original_image.width - 1, original_image.height - 1));
						}

						crop_rect = cvRect(prop_box.left, prop_box.top, prop_box.width(), prop_box.height());
					}
					else {
						crop_rect = cvRect(0, 0, original_image.width, original_image.height); // full image for images without boxes
						crop_rect2 = crop_rect;
					}
				}
				else if (!entry_config.exist("crop_ratio_lowerbound")
					|| !entry_config.exist("crop_ratio_upperbound")) { // fixed size cropping
					scale_selected = entry_config.get<float>("crop_ratio");
					crop_patch(crop_rect, im_temp, scale_selected, croptype, crop_position);
				}
				else if (entry_config.get<string>("scale_jitter_type") == "UniAreaV2") { 
					// UniAreaV2 jitter is different from others. So process it individually
					float up_ratio = entry_config.get<float>("crop_ratio_upperbound");
					float low_ratio = entry_config.get<float>("crop_ratio_lowerbound");

					float area = (float)(im_temp.cols * im_temp.rows);

					int attempt = 0;
					int max_attempt = 10;
					while (attempt < max_attempt){
						float target_area = (float)random_helper::uniform_real(low_ratio, up_ratio) * area;

						float aspect_ratio = (float)random_helper::uniform_real(3.f / 4.f, 4.f / 3.f);

						int target_w = (int)(sqrtf(target_area * aspect_ratio) + .5f);
						int target_h = (int)(sqrtf(target_area / aspect_ratio) + .5f);

						if ((float)random_helper::uniform_real(0.f, 1.f) < .5f) std::swap(target_w, target_h);

						if (target_w <= im_temp.cols && target_h <= im_temp.rows) {
							crop_patch_nonuniform(crop_rect, im_temp, target_w, target_h, croptype, crop_position);
							break;
						}
						attempt++;
					} //while

					if (attempt == max_attempt) crop_patch(crop_rect, im_temp, 1.0f, croptype, crop_position);
				}
				else { // jitting
					float up_scale = entry_config.get<float>("crop_ratio_upperbound");
					float low_scale = entry_config.get<float>("crop_ratio_lowerbound");

					CHECK_LE(low_scale, up_scale);
					CHECK_GT(low_scale, 0);
					CHECK_GT(up_scale, 0);

					const string scale_jitter_type = entry_config.get<string>("scale_jitter_type");
					float cur_scale = scale_selected;

					if (scale_selected == 0) { // for each trial, we only select scale once, while crop multiple times from different positions
						if (scale_jitter_type == "UniRatio") {
							cur_scale = (float)random_helper::uniform_real(low_scale, up_scale);
						}
						else if (scale_jitter_type == "UniLength") {
							float size_upper = (float)batch_img_size / low_scale;
							float size_lower = (float)batch_img_size / up_scale;
							float cur_size = (float)random_helper::uniform_real(size_lower, size_upper);
							cur_scale = (float)batch_img_size / cur_size;
						}
						else if (scale_jitter_type == "UniArea") {
							float area_upper = (float)batch_img_size / low_scale;
							area_upper *= area_upper;
							float area_lower = (float)batch_img_size / up_scale;
							area_lower *= area_lower;
							float cur_area = (float)random_helper::uniform_real(area_lower, area_upper);
							cur_scale = (float)batch_img_size / sqrtf(cur_area);
						}
						else {
							LOG(FATAL) << "Unknown scale jitter type";
						}

						scale_selected = cur_scale;
					}

					crop_patch(crop_rect, im_temp, cur_scale, croptype, crop_position);
				}

				if (entry_config.exist("crop_preference") || entry_config.exist("random_crop_overlap_threshold")) {

					string crop_preference = entry_config.get<string>("crop_preference", false, "IoCThres");
					float thres = entry_config.get<float>("random_crop_overlap_threshold", false, 0.f);
					if (crop_preference == "IoCThres") {
						CHECK(!provider_config.get<string>("metadata_file").empty());

						if (thres > 0 && bboxes.size() > 0) {
							Box current_crop_box(crop_rect);
							if (bFlipLR) {
								current_crop_box = current_crop_box.flip(Box(0, 0, original_image.width - 1, original_image.height - 1, -1));
							}

							float max_crop_overlap = -1;
							for (int i = 0; i < bboxes.size(); i++) {
								float ioc = (float)current_crop_box.IoC(bboxes[i]);
								max_crop_overlap = std::max(max_crop_overlap, ioc);
							}

							if (max_crop_overlap < thres) continue; // try to crop again
						}
					}
					else if (crop_preference == "IoUMax") {
						NOT_IMPLEMENTED;
					}
					else if (crop_preference == "IoUThres") {
						CHECK(!provider_config.get<string>("metadata_file").empty());

						if (thres > 0 && bboxes.size() > 0) {
							Box current_crop_box(crop_rect);
							if (bFlipLR) {
								current_crop_box = current_crop_box.flip(Box(0, 0, original_image.width - 1, original_image.height - 1, -1));
							}

							float max_crop_overlap = -1;
							for (int i = 0; i < bboxes.size(); i++) {
								float iou = (float)current_crop_box.IoU(bboxes[i]);
								max_crop_overlap = std::max(max_crop_overlap, iou);
							}

							if (max_crop_overlap < thres) continue; // try to crop again
						}
					}
					else if (crop_preference == "AnchorMaxThres") {
						CHECK(!provider_config.get<string>("metadata_file").empty());

						if (thres > 0 && bboxes.size() > 0) {
							Box current_crop_box(crop_rect);
							if (bFlipLR) {
								current_crop_box = current_crop_box.flip(Box(0, 0, original_image.width - 1, original_image.height - 1, -1));
							}

							data_utils::box_anchor_transform trans_helper(current_crop_box, original_image.width, original_image.height, entry_config);

							bool pass = false;

							for (int i = 0; i < trans_helper.anchors().size(); i++) {
								data_type iou;
								trans_helper.get_maxiou_box(i, bboxes, &iou);
								if (iou >= thres) {
									pass = true;
									break;
								}
							}

							if (!pass) continue; // try to crop again
						}
					}
				}

				break; // in most cases, we do not need try cropping so many times
			}

			if (max_IoU_value > 0) { // used for max IoU crop
				crop_rect = max_IoU_crop;
			}

			if (croptype == data_utils::CropType_FullView) {
				// for full-view crop, the crop_rect is only a baseline for anchors, not the actual crop box
				cv::Rect actual_crop_rect(
					crop_rect.x, crop_rect.y, im_temp.cols - 2 * crop_rect.x, im_temp.rows - 2 * crop_rect.y
					);

				crop_img(im_cropped, im_temp, actual_crop_rect, bFlipLR);
				// HACK:
				// why can we use bFlipLR here safety? because even though we may need to crop the central rect
				// when for large image width or height, since the rect is always lies in the center of raw image,
				// so flip the rect equals to flip the raw image, along with the bounding boxes and the "baseline" rect
			}
			else {
				crop_img(im_cropped, im_temp, crop_rect, bFlipLR);
			}

			// 3. trans to float 
			processed_image.data.reset(new cv::Mat());
			im_cropped.convertTo(*processed_image.data, CV_32F);

			// 4. resize im size to this->layer_param_.batchimgsize()
			//   4.1 select interpolation algorithm
			int interpolation = cv::INTER_LINEAR;
			vector<string> interpolations = entry_config.get_array<string>("interpolation", false);
			if (interpolations.size() > 0) {
				string sel = interpolations[0];

				int n = (int)interpolations.size();
				if (n > 1) {
					// random selection
					int idx = rand() % n;
					sel = interpolations[idx];
				}

				if (sel == "Bilinea") {
					interpolation = cv::INTER_LINEAR;
				}
				else if (sel == "Bicubic") {
					interpolation = cv::INTER_CUBIC;
				}
				else if (sel == "Lanczos") {
					interpolation = cv::INTER_LANCZOS4;
				}
				else {
					CHECK(false) << "Invalid interpolation type: " << sel;
				}
			}

			//   4.2 resize
			if (croptype == data_utils::CropType_FullView) {
				// here, crop_rect.width == crop_rect.height == min(width, height) * scale
				int actual_img_width = (int)((double)batch_img_size / crop_rect.width * processed_image.data->cols + 0.5);
				int actual_img_height = (int)((double)batch_img_size / crop_rect.height * processed_image.data->rows + 0.5);

				cv::resize(*processed_image.data, *processed_image.data, cv::Size(actual_img_width, actual_img_height), 0.0, 0.0, interpolation);
			}
			else {
				cv::resize(*processed_image.data, *processed_image.data, cv::Size(batch_img_size, batch_img_size), 0.0, 0.0, interpolation);
			}

			// 5. data_extend
			// removed from current version

			// 6. modify picture pixels respected to color KL matrix
			if (original_image.additional_data.count("kl_info")) {
				float shift[3];
				data_utils::kl_info kl = boost::any_cast<data_utils::kl_info>(original_image.additional_data.find("kl_info")->second);
				auto &im = *processed_image.data;
				data_utils::random_color_shift(shift, kl);
				CHECK_EQ(im.channels(), 3);

				//int pixels = batchimgsize * batchimgsize;
				int pixels = im.rows * im.cols;
				int nChannel = im.channels();
				for (int y = 0; y < im.rows; y++) {
					float *pImg = (float*)im.ptr(y);
					for (int x = 0; x < im.cols; x++) {
						pImg[nChannel*x] += shift[2];
						pImg[nChannel*x + 1] += shift[1];
						pImg[nChannel*x + 2] += shift[0];
					}
				}
			}

			// 7. minus mean 
			if (!data_mean_.empty()) {
				auto &im = *processed_image.data;
				CHECK_EQ(im.channels(), data_mean_.channels());
				if (im.rows == data_mean_.rows && im.cols == data_mean_.cols) {
					im = im - data_mean_;
				}
				else {
					cv::Mat resized_mean;
					cv::resize(data_mean_, resized_mean, cv::Size(im.cols, im.rows), 0.0, 0.0, interpolation);
					im = im - resized_mean;
				}
			}
		}

		void image_entry::crop_patch_nonuniform(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, int target_w, int target_h,
			data_utils::crop_type croptype, __in_opt data_utils::crop_position cropposition) {

			int height = matRawImg.rows, width = matRawImg.cols;

			if (target_w <= 0 || target_w > width || target_h <= 0 || target_h > height) {
				LOG(FATAL) << "CropImg::Invalid target_w target_h!";
			}

			int half_x = (width - target_w) / 2;
			int half_y = (height - target_h) / 2;

			int x_off = -1, y_off = -1;

			switch (croptype) {
			case data_utils::CropType_Random:
				y_off = random_helper::uniform_int() % (height - target_h + 1);
				x_off = random_helper::uniform_int() % (width - target_w + 1);
				break;

			default:
				LOG(FATAL) << "CropImg::Undefined Crop Type!" << std::endl;
				break;
			}

			cropRect = cv::Rect(x_off, y_off, target_w, target_h);
		}

		void image_entry::crop_patch(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, float fCropRatio,
			data_utils::crop_type croptype, __in_opt data_utils::crop_position cropposition) {

			if (fCropRatio <= 0 || fCropRatio > 1) {
				LOG(FATAL) << "CropImg::Invalid fCropRatio!";
			}

			int height = matRawImg.rows, width = matRawImg.cols;
			int crop_size = (int)(std::min(height, width) * fCropRatio);
			int half_x = (width - crop_size) / 2;
			int half_y = (height - crop_size) / 2;

			int crop_size_x = crop_size, crop_size_y = crop_size; // default crop size
			int x_off = -1, y_off = -1;

			switch (croptype) {
			case data_utils::CropType_Random:
				y_off = random_helper::uniform_int() % (height - crop_size + 1);
				x_off = random_helper::uniform_int() % (width - crop_size + 1);
				break;

			case data_utils::CropType_10View: {
				const int x_offs[data_utils::CropPosition_End - data_utils::CropPosition_Begin] = { 0, 2 * half_x, 0, 2 * half_x, half_x, half_x, 2 * half_x, half_x, 0, half_x };
				const int y_offs[data_utils::CropPosition_End - data_utils::CropPosition_Begin] = { 0, 0, 2 * half_y, 2 * half_y, half_y, 0, half_y, 2 * half_y, half_y, half_y };

				y_off = y_offs[cropposition - data_utils::CropPosition_Begin - 1];
				x_off = x_offs[cropposition - data_utils::CropPosition_Begin - 1];
				break;
			}

			case data_utils::CropType_Center:
				y_off = half_y;
				x_off = half_x;
				break;

			case data_utils::CropType_18View: {
				const int base_x = 0, base_y = 0;
				const int x_offs[data_utils::CropPosition_End - data_utils::CropPosition_Begin] = { base_x, 2 * half_x - base_x, base_x, 2 * half_x - base_x, half_x, half_x, 2 * half_x - base_x, half_x, base_x, half_x, base_x, 2 * half_x - base_x, base_x, 2 * half_x - base_x, half_x, 2 * half_x - base_x, half_x, base_x };
				const int y_offs[data_utils::CropPosition_End - data_utils::CropPosition_Begin] = { base_y, base_y, 2 * half_y - base_y, 2 * half_y - base_y, half_y, base_y, half_y, 2 * half_y - base_y, half_y, half_y, base_y, base_y, 2 * half_y - base_y, 2 * half_y - base_y, base_y, half_y, 2 * half_y - base_y, half_y };

				y_off = y_offs[cropposition - data_utils::CropPosition_Begin - 1];
				x_off = x_offs[cropposition - data_utils::CropPosition_Begin - 1];
				break;
			}

			case data_utils::CropType_MultiCrop: {
				int w_segs = config_.get<int>("crop_dim1_segs");
				int h_segs = config_.get<int>("crop_dim2_segs");
				CHECK_GT(w_segs, 0);
				CHECK_GT(h_segs, 0);

				if (w_segs < h_segs && width > height) std::swap(w_segs, h_segs);
				else if (w_segs > h_segs && width < height) std::swap(w_segs, h_segs);

				int crop_index = (int)(cropposition - data_utils::CropPosition_Begin - 1);
				if (crop_index >= w_segs * h_segs) crop_index %= w_segs * h_segs;
				int w_index = crop_index % w_segs, h_index = crop_index / w_segs;

				if (w_segs == 1) x_off = half_x;
				else x_off = (int)((double)(width - crop_size) / (w_segs - 1) * w_index + 0.5);

				if (h_segs == 1) y_off = half_y;
				else y_off = (int)((double)(height - crop_size) / (h_segs - 1) * h_index + 0.5);

				break;
			}
			case data_utils::CropType_FullView: {
				// for full-view crop, crop box only acts as a "baseline" for anchors
				// not the actual crop box!!

				const int MAX_ACTUAL_EDGE_LENGTH = 1400;
				int actual_crop_size = config_.get<int>("batch_img_size");
				//int actual_img_width = (int)((double)actual_crop_size / crop_size * width + 0.5);
				//int actual_img_height = (int)((double)actual_crop_size / crop_size * height + 0.5);

				int max_img_edge_length = (int)((double)crop_size / actual_crop_size * MAX_ACTUAL_EDGE_LENGTH + 0.5);

				CHECK(width <= max_img_edge_length || height <= max_img_edge_length);
				if (width <= max_img_edge_length && height <= max_img_edge_length) {
					// default baseline (left-top corner)
					x_off = 0;
					y_off = 0;
				}
				else if (width > max_img_edge_length) {
					// too wide. crop center
					x_off = (width - max_img_edge_length) / 2;
					y_off = 0;
				}
				else if (height > max_img_edge_length) {
					// too tall, crop center
					x_off = 0;
					y_off = (height - max_img_edge_length) / 2;
				}
				else {
					LOG(FATAL) << "Bug";
				}
				break;
			}

			default:
				LOG(FATAL) << "CropImg::Undefined Crop Type!" << std::endl;
				break;
			}

			cropRect = cv::Rect(x_off, y_off, crop_size_x, crop_size_y);
		}

		void image_entry::crop_img(__out cv::Mat &matOutImg, __in const cv::Mat &matRawImg, __in cv::Rect &cropRect, bool bFlipLR) {
			matOutImg = matRawImg(cropRect).clone();

			if (bFlipLR) cv::flip(matOutImg, matOutImg, 1);
		}
	}
}