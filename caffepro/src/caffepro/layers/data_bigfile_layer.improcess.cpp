
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/random_helper.h>
#include <caffepro/utils/data_utils/box.h>
#include <caffepro/utils/data_utils/box_trans.h>

using std::vector;
using std::string;
using std::map;

namespace caffepro {

	using data_utils::crop_type;
	using data_utils::Box;
	using data_utils::BoxF;
	using data_utils::box_anchor_transform;

	void data_bigfile_layer::prepare_one_image(__out cv::Mat &im, __in data_utils::raw_picture &raw_data,
		__in int itemid, __in int total_channels, __in_opt const cv::Mat meanIm, int view_id) {
		cv::Mat im_temp, im_cropped;

		// 1. decode image
		if (total_channels == 3) {
			data_utils::rawpicture_to_im(im_temp, raw_data);
		}
		else {
			rawpicture_to_im_adapt(im_temp, raw_data);
		}

		if (layer_param_.data_bigfile_param().has_metadata_file()) {
			CHECK_EQ(raw_data.width, im_temp.cols);
			CHECK_EQ(raw_data.height, im_temp.rows);
		}

		// 2. crop image
		int iViewIdx = view_id == -1 ? raw_data.get_current_view_idx() : view_id;
		bool bFlipLR = false;
		crop_type croptype = crop_type(this->layer_param_.data_bigfile_param().crop_type());
		switch (croptype) {
		case data_utils::CropType_Random:
			bFlipLR = random_helper::uniform_int() % 2 != 0;
			break;
		case data_utils::CropType_10View:
			bFlipLR = iViewIdx >= 5 ? true : false;
			break;
		case data_utils::CropType_18View: {
											  const bool fl[] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 };
											  bFlipLR = fl[iViewIdx];
											  break;
		}
		case data_utils::CropType_MultiCrop: {
												 int segs = layer_param_.data_bigfile_param().crop_dim1_segs() * layer_param_.data_bigfile_param().crop_dim2_segs();
												 bFlipLR = (iViewIdx / segs % 2 == 1);
												 break;
		}
		case data_utils::CropType_PropWarp:
			if (context_->get_phase() == caffepro_context::TRAIN) {
				bFlipLR = random_helper::uniform_int() % 2 != 0;
			}
			break;
		case data_utils::CropType_FullView:
			bFlipLR = iViewIdx % 2 != 0;
			break;
		}

		data_utils::crop_position crop_position = data_utils::crop_position(iViewIdx + data_utils::CropPosition_Begin + 1);

		vector<Box> bboxes = raw_data.bounding_boxes;
		if (bFlipLR) { // box positions should also be flipped
			Box image_box(0, 0, raw_data.width - 1, raw_data.height - 1, -1);

			for (int i = 0; i < (int)bboxes.size(); i++) {
				bboxes[i] = bboxes[i].flip(image_box);
			}
		}

		cv::Rect crop_rect;
		cv::Rect crop_rect2; // for RCNN, since pad may applied, we should backup the original crop rect
		float scale_selected = 0;

		cv::Rect max_IoU_crop;
		float max_IoU_value = 0;

		int max_crop_turn = 50;
		while (max_crop_turn--) {
			if (croptype == data_utils::CropType_PropWarp) {
				scale_selected = 1;
				if (raw_data.prop_boxes.size() > 0 && raw_data.prop_boxes[iViewIdx % raw_data.prop_boxes.size()].valid()) {
					Box prop_box = raw_data.prop_boxes[iViewIdx % raw_data.prop_boxes.size()];
					crop_rect2 = cvRect(prop_box.left, prop_box.top, prop_box.width(), prop_box.height());	// backup original box

					if (this->layer_param_.data_bigfile_param().has_rcnn_pad()) {
						float pad_ratio = this->layer_param_.data_bigfile_param().rcnn_pad();
						int pad_width = (int)((prop_box.width() * pad_ratio) / 2), pad_height = (int)((prop_box.height() * pad_ratio) / 2);
						prop_box.left -= pad_width;
						prop_box.top -= pad_height;
						prop_box.right += pad_width;
						prop_box.bottom += pad_height;
						prop_box = prop_box.intersect(Box(0, 0, raw_data.width - 1, raw_data.height - 1));
					}

					crop_rect = cvRect(prop_box.left, prop_box.top, prop_box.width(), prop_box.height());
				}
				else {
					crop_rect = cvRect(0, 0, raw_data.width, raw_data.height); // full image for images without boxes
					crop_rect2 = crop_rect;
				}
			}
			else if (!this->layer_param_.data_bigfile_param().has_crop_ratio_lowerbound()
				|| !this->layer_param_.data_bigfile_param().has_crop_ratio_upperbound()) { // fixed size cropping
				crop_patch(crop_rect, im_temp, this->layer_param_.data_bigfile_param().crop_ratio(), croptype, crop_position);
				scale_selected = this->layer_param_.data_bigfile_param().crop_ratio();
			}
			else if (this->layer_param_.data_bigfile_param().scale_jitter_type() == DataParameter_BigFile_ScaleJitterType_UniAreaV2) {
				float up_ratio = this->layer_param_.data_bigfile_param().crop_ratio_upperbound();
				float low_ratio = this->layer_param_.data_bigfile_param().crop_ratio_lowerbound();

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
				float up_scale = this->layer_param_.data_bigfile_param().crop_ratio_upperbound();
				float low_scale = this->layer_param_.data_bigfile_param().crop_ratio_lowerbound();

				CHECK_LE(low_scale, up_scale);
				CHECK_GT(low_scale, 0);
				CHECK_GT(up_scale, 0);

				auto scale_jitter_type = this->layer_param_.data_bigfile_param().scale_jitter_type();
				float cur_scale = scale_selected;

				if (scale_selected == 0) { // for each trial, we only select scale once, while crop multiple times from different positions
					if (scale_jitter_type == DataParameter_BigFile_ScaleJitterType_UniRatio) {
						cur_scale = (float)random_helper::uniform_real(low_scale, up_scale);
					}
					else if (scale_jitter_type == DataParameter_BigFile_ScaleJitterType_UniLength) {
						const int batchimgsize = this->layer_param_.data_bigfile_param().batch_img_size();
						float size_upper = (float)batchimgsize / low_scale;
						float size_lower = (float)batchimgsize / up_scale;
						float cur_size = (float)random_helper::uniform_real(size_lower, size_upper);
						cur_scale = (float)batchimgsize / cur_size;
					}
					else if (scale_jitter_type == DataParameter_BigFile_ScaleJitterType_UniArea) {
						const int batchimgsize = this->layer_param_.data_bigfile_param().batch_img_size();
						float area_upper = (float)batchimgsize / low_scale;
						area_upper *= area_upper;
						float area_lower = (float)batchimgsize / up_scale;
						area_lower *= area_lower;
						float cur_area = (float)random_helper::uniform_real(area_lower, area_upper);
						cur_scale = (float)batchimgsize / sqrtf(cur_area);
					}
					else {
						LOG(FATAL) << "Unknown scale jitter type";
					}

					scale_selected = cur_scale;
				}

				crop_patch(crop_rect, im_temp, cur_scale, croptype, crop_position);
			}

			if (this->layer_param_.data_bigfile_param().has_crop_preference()
				|| this->layer_param_.data_bigfile_param().has_random_crop_overlap_threshold()) {
				if (this->layer_param_.data_bigfile_param().crop_preference() == DataParameter_BigFile_CropPreference_IoCThres) {
					float thres = this->layer_param_.data_bigfile_param().random_crop_overlap_threshold();

					CHECK(this->layer_param_.data_bigfile_param().has_metadata_file());

					if (thres > 0 && bboxes.size() > 0) {
						Box current_crop_box(crop_rect);
						if (bFlipLR) {
							current_crop_box = current_crop_box.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
						}

						float max_crop_overlap = -1;
						for (int i = 0; i < bboxes.size(); i++) {
							float ioc = (float)current_crop_box.IoC(bboxes[i]);
							max_crop_overlap = std::max(max_crop_overlap, ioc);
						}

						if (max_crop_overlap < thres) continue; // try to crop again
					}
				}
				else if (this->layer_param_.data_bigfile_param().crop_preference() == DataParameter_BigFile_CropPreference_IoUMax) {
					NOT_IMPLEMENTED;
				}
				else if (this->layer_param_.data_bigfile_param().crop_preference() == DataParameter_BigFile_CropPreference_IoUThres) {
					CHECK(this->layer_param_.data_bigfile_param().has_random_crop_overlap_threshold());

					float thres = this->layer_param_.data_bigfile_param().random_crop_overlap_threshold();

					CHECK(this->layer_param_.data_bigfile_param().has_metadata_file());
					//CHECK_EQ(Caffe::phase(), Caffe::TRAIN);

					if (thres > 0 && bboxes.size() > 0) {
						Box current_crop_box(crop_rect);
						if (bFlipLR) {
							current_crop_box = current_crop_box.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
						}

						float max_crop_overlap = -1;
						for (int i = 0; i < bboxes.size(); i++) {
							float iou = (float)current_crop_box.IoU(bboxes[i]);
							max_crop_overlap = std::max(max_crop_overlap, iou);
						}

						if (max_crop_overlap < thres) continue; // try to crop again
					}
				}
				else if (this->layer_param_.data_bigfile_param().crop_preference() == DataParameter_BigFile_CropPreference_AnchorMaxThres) {
					CHECK(this->layer_param_.data_bigfile_param().has_random_crop_overlap_threshold());
					float thres = this->layer_param_.data_bigfile_param().random_crop_overlap_threshold();
					CHECK(this->layer_param_.data_bigfile_param().has_metadata_file());
					CHECK(config_.valid()) << "You must provide config file for the net";

					if (thres > 0 && bboxes.size() > 0) {
						Box current_crop_box(crop_rect);
						if (bFlipLR) {
							current_crop_box = current_crop_box.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
						}

						box_anchor_transform trans_helper(current_crop_box, raw_data.width, raw_data.height, config_);

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
		im_cropped.convertTo(im, CV_32F);

		// 4. resize im size to this->layer_param_.batchimgsize()
		//   4.1 select interpolation algorithm
		int interpolation = cv::INTER_LINEAR;
		if (this->layer_param_.data_bigfile_param().interpolation_size() > 0) {
			auto &big_file_param = this->layer_param_.data_bigfile_param();
			auto sel = big_file_param.interpolation(0);

			int n = big_file_param.interpolation_size();
			if (n > 1) {
				// random selection
				int idx = rand() % n;
				sel = big_file_param.interpolation(idx);
			}

			switch (sel) {
			case DataParameter_BigFile_InterpolationType_Bilinear:
				interpolation = cv::INTER_LINEAR;
				break;

			case DataParameter_BigFile_InterpolationType_Bicubic:
				interpolation = cv::INTER_CUBIC;
				break;

			case DataParameter_BigFile_InterpolationType_Lanczos:
				interpolation = cv::INTER_LANCZOS4;
				break;

			default:
				CHECK(false) << "Invalid interpolation type: " << sel;
				break;
			}
		}

		//   4.2 resize
		const int batchimgsize = this->layer_param_.data_bigfile_param().batch_img_size();
		if (croptype == data_utils::CropType_FullView) {
			// here, crop_rect.width == crop_rect.height == min(width, height) * scale
			int actual_img_width = (int)((double)batchimgsize / crop_rect.width * im.cols + 0.5);
			int actual_img_height = (int)((double)batchimgsize / crop_rect.height * im.rows + 0.5);

			cv::resize(im, im, cv::Size(actual_img_width, actual_img_height), 0.0, 0.0, interpolation);
		}
		else {
			cv::resize(im, im, cv::Size(batchimgsize, batchimgsize), 0.0, 0.0, interpolation);
		}

		// 5. data_extend
		// removed from current version

		// 6. modify picture pixels respected to color KL matrix
		float shift[3];
		if (data_utils::random_color_shift(shift, database_.class_info, raw_data)) {
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
		if (!meanIm.empty()) {
			CHECK_EQ(im.channels(), meanIm.channels());
			if (im.rows == meanIm.rows && im.cols == meanIm.cols) {
				im = im - meanIm;
			}
			else {
				cv::Mat resized_mean;
				cv::resize(meanIm, resized_mean, cv::Size(im.cols, im.rows), 0.0, 0.0, interpolation);
				im = im - resized_mean;
			}
		}

		auto &processing_imgs = prefetch_.prefetch_batch_state.processing_imgs;
		processing_imgs[itemid] = raw_data;

		// 8. extra data processer
		for (int p = 0; p < layer_param_.data_bigfile_param().additional_data_processer_size(); p++) {
			auto &cur_processer = layer_param_.data_bigfile_param().additional_data_processer(p);

			if (cur_processer.processer_type() == "rcnn_label") {
				Box current_crop_box(crop_rect);		// crop box (may have pad)
				Box current_crop_box_raw(crop_rect2);	// original crop box (no pad)
				if (bFlipLR) {
					// the pos of crop_rect is not moved
					// because the pipeline is crop first, then flip the crop image
					// however, when generating GT boxes, we must flip the original image first, then find the correct position
					current_crop_box = current_crop_box.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
					current_crop_box_raw = current_crop_box_raw.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
				}

				int output_idx = cur_processer.binding_output_index(0);

				double max_iou = -1;
				for (auto iter = bboxes.begin(); iter != bboxes.end(); ++iter) {
					max_iou = std::max(max_iou, current_crop_box_raw.IoU(*iter));	// use original box to calculate IoU
				}

				int rcnn_label = (int)(max_iou > 0.5);
				prefetch_.extra_data[output_idx - 1 - 1]->get(0)->mutable_cpu_data()[itemid] = (data_type)rcnn_label;

				processing_imgs[itemid].bounding_boxes = bboxes;
				processing_imgs[itemid].flipped = bFlipLR;
				processing_imgs[itemid].crop_box = current_crop_box;	// here we use crop box, not original box
			}
			else if (cur_processer.processer_type() == "bounding_box") {
				map<string, float> params;
				int cur_thres = 0;
				for (int i = 1; i < cur_processer.method_size(); i++) {
					if (cur_processer.method(i) == "accept_rate") {
						CHECK_LT(cur_thres, cur_processer.threshold_size());
						params[cur_processer.method(i)] = cur_processer.threshold(cur_thres);
						cur_thres++;
					}
					else {
						LOG(FATAL) << "Unknown method: " << cur_processer.method(i);
					}
				}

				BoxF trans_gtbox = BoxF::get_invalid_box();
				BoxF trans_imagebox = BoxF::get_invalid_box();

				if (bboxes.size() > 0) {
					Box current_crop_box(crop_rect);
					if (bFlipLR) {
						// the pos of crop_rect is not moved
						// because the pipeline is crop first, then flip the crop image
						// however, when generating GT boxes, we must flip the original image first, then find the correct position
						current_crop_box = current_crop_box.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
					}

					// in current framework, cost layer do not relay on this value any more
					// so naive select the first one
					int sel = -1;
					if (bboxes.size() > 0) sel = 0;

					if (sel >= 0) {
						// do not execute transform here
						Box gtbox = bboxes[sel];
						trans_gtbox.from(gtbox);
						trans_imagebox = BoxF(0, 0, (float)(raw_data.width - 1), (float)(raw_data.height - 1), -1);
					}
					else {
						CHECK_EQ(context_->get_phase(), caffepro_context::TRAIN);
					}
				}

				int output_idx = cur_processer.binding_output_index(0);
				data_type *data = prefetch_.extra_data[output_idx - 1 - 1]->get(0)->mutable_cpu_data() + itemid * 8;
				trans_gtbox.fill(data);
				trans_imagebox.fill(data + 4);

				processing_imgs[itemid].bounding_boxes = bboxes;
				processing_imgs[itemid].flipped = bFlipLR;
				if (trans_gtbox.valid()) {
					Box current_crop_box(crop_rect);

					// for full-view test, we should not flip the crop_box
					// because it is not actual crop box, but a baseline for anchors
					// we only need to flip the ground-truth boxes
					if (bFlipLR && croptype != data_utils::CropType_FullView) {
						current_crop_box = current_crop_box.flip(Box(0, 0, raw_data.width - 1, raw_data.height - 1, -1));
					}
					processing_imgs[itemid].crop_box = current_crop_box;
				}
				else {
					processing_imgs[itemid].crop_box = Box::get_invalid_box();
				}
			}
			else {
				LOG(FATAL) << "Unknown data processer type: " << cur_processer.processer_type();
			}
		}
	}

	void data_bigfile_layer::crop_patch(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, float fCropRatio,
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

		case data_utils::CropType_10View: 
		{
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

		case data_utils::CropType_18View: 
		{
			const int base_x = 0, base_y = 0;
			const int x_offs[data_utils::CropPosition_End - data_utils::CropPosition_Begin] = { base_x, 2 * half_x - base_x, base_x, 2 * half_x - base_x, half_x, half_x, 2 * half_x - base_x, half_x, base_x, half_x, base_x, 2 * half_x - base_x, base_x, 2 * half_x - base_x, half_x, 2 * half_x - base_x, half_x, base_x };
			const int y_offs[data_utils::CropPosition_End - data_utils::CropPosition_Begin] = { base_y, base_y, 2 * half_y - base_y, 2 * half_y - base_y, half_y, base_y, half_y, 2 * half_y - base_y, half_y, half_y, base_y, base_y, 2 * half_y - base_y, 2 * half_y - base_y, base_y, half_y, 2 * half_y - base_y, half_y };

			y_off = y_offs[cropposition - data_utils::CropPosition_Begin - 1];
			x_off = x_offs[cropposition - data_utils::CropPosition_Begin - 1];
			break;
		}

		case data_utils::CropType_MultiCrop:
		{
			int w_segs = layer_param_.data_bigfile_param().crop_dim1_segs();
			int h_segs = layer_param_.data_bigfile_param().crop_dim2_segs();
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
			int actual_crop_size = layer_param_.data_bigfile_param().batch_img_size();
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

	void data_bigfile_layer::crop_img(__out cv::Mat &matOutImg, __in const cv::Mat &matRawImg, __in cv::Rect &cropRect, bool bFlipLR) {
		matOutImg = matRawImg(cropRect).clone();

		if (bFlipLR) cv::flip(matOutImg, matOutImg, 1);
	}

	void data_bigfile_layer::crop_patch_nonuniform(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, int target_w, int target_h,
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
}