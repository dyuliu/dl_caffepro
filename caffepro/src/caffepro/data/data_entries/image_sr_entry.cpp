
#include <caffepro/data/data_entries/image_sr_entry.h>
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

		image_sr_entry::image_sr_entry(data_provider &provider, const std::string &name)
			: data_entry(provider, name) {
			type_ = "image_sr";
		}

		image_sr_entry::~image_sr_entry() {
			// nothing to do
		}

		void image_sr_entry::init() {
			data_entry::init();

			split_minibatch_ = provider_.data_provider_config().get_array<int>("split_minibatch", true);
			split_gpu_id_ = provider_.data_provider_config().get_array<int>("split_gpu_id", true);

			const int batchsize = provider_.data_provider_config().get<int>("batch_size");
			const int batch_img_size = config_.get<int>("batch_img_size");
			const int channel_num_total = config_.get<int>("channel_num");

			auto_init_buffer(0, batchsize, channel_num_total, batch_img_size, batch_img_size, false); // degrade image
			
			caffepro_config_reader &entry_config = config_;
			const int pad_inner = entry_config.get<int>("pad_inner");
			auto_init_buffer(1, batchsize, channel_num_total, batch_img_size - 2 * pad_inner, batch_img_size - 2 * pad_inner, false); // clean image


			const int sr_ratio = config_.get<int>("super_resolution_ratio");
			if (sr_ratio == 1)
				LOG(ERROR) << "Warning: using denoising mode.";
			else
				LOG(ERROR) << "Warning: using super-resolution mode.";
		}

		void image_sr_entry::prepare(batch_descriptor &batch) {
			// parameters
			caffepro_config_reader &provider_config = provider_.data_provider_config();
			caffepro_config_reader &entry_config = config_;

			const int batchsize = provider_config.get<int>("batch_size");
			const int channel_num_total = config_.get<int>("channel_num");
			crop_type croptype = crop_type(provider_config.get<int>("crop_type"));
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

			// degrade images
			const int sr_ratio = config_.get<int>("super_resolution_ratio");
#pragma omp parallel for schedule(dynamic)
			for (int itemid = 0; itemid < batchsize; itemid++) {
				auto &item = batch.batch_data[itemid];
				if (sr_ratio == 1)
					addnoise_one_image(*item.processed_data);
				else
					degrade_one_image(*item.processed_data);
				CHECK_EQ(item.processed_data->data->channels(), channel_num_total) << "Prepared image.channels() != sum(channel_num)";
			}
			
			// write back images
			// currently, only full-view crop is not fixed size
			bool fixed_size = (croptype != data_utils::CropType_FullView);
			write_batch_to_blob(batch, prefetch_buffer_[1], fixed_size); // clean image in #id [1]

#pragma omp parallel for schedule(dynamic)
			for (int itemid = 0; itemid < batchsize; itemid++) {
				auto &item = batch.batch_data[itemid];

				auto &im1 = *item.processed_data->data;
				cv::Mat im = boost::any_cast<cv::Mat>(item.processed_data->additional_data["im_degraded"]);
				*item.processed_data->data = im.clone();

				CHECK_EQ(item.processed_data->data->channels(), channel_num_total) << "Prepared image.channels() != sum(channel_num)";
			}
			write_batch_to_blob(batch, prefetch_buffer_[0], fixed_size); // degraded image in #id [0]
		}

		void image_sr_entry::prepare_one_image(const data_container &original_image, data_container &processed_image) {
			caffepro_config_reader &provider_config = provider_.data_provider_config();
			caffepro_config_reader &entry_config = config_;

			cv::Mat im_temp, im_temp_lowres;

			const int channel_num_total = config_.get<int>("channel_num");

			// 1. decode image
			if (channel_num_total == 1)
				im_temp = cv::imdecode(*original_image.data, CV_LOAD_IMAGE_GRAYSCALE);
			else
				im_temp = cv::imdecode(*original_image.data, CV_LOAD_IMAGE_COLOR);

			// 1.1 split into two images
			CHECK_EQ(im_temp.cols % 2, 0);
			
			cv::Rect rc1(0, 0, im_temp.cols / 2, im_temp.rows);
			cv::Rect rc2(im_temp.cols / 2, 0, im_temp.cols / 2, im_temp.rows);
			
			im_temp_lowres = im_temp(rc2).clone();
			im_temp = im_temp(rc1).clone();

			// 2. crop image
			int iViewIdx = -1;
			if (processed_image.additional_data.count("view_id")) {
				iViewIdx = boost::any_cast<int>(processed_image.additional_data["view_id"]);
			}
			bool bFlipLR = false;
			crop_type croptype = crop_type(provider_config.get<int>("crop_type"));
			switch (croptype) {
			case data_utils::CropType_Random:
				bFlipLR = random_helper::uniform_int() % 2 != 0;
				break;
			case data_utils::CropType_10View:
				if (iViewIdx < 0) iViewIdx = random_helper::uniform_int(0, 9);
				bFlipLR = iViewIdx >= 5 ? true : false;
				break;
			case data_utils::CropType_FullView:
				break;
			default:
				LOG(FATAL) << "Unsupport Crop Type!" << std::endl;
				break;
			}

			data_utils::crop_position crop_position = data_utils::crop_position(iViewIdx + data_utils::CropPosition_Begin + 1);

			int batch_img_size = entry_config.get<int>("batch_img_size");

			cv::Rect crop_rect;
			float scale_selected = 0;

			if (croptype != data_utils::CropType_FullView) {
				if (!entry_config.exist("crop_ratio_lowerbound")
					|| !entry_config.exist("crop_ratio_upperbound")) { // fixed size cropping
					scale_selected = entry_config.get<float>("crop_ratio");

					if (scale_selected < 0) { // do not scale
						crop_patch_not_scaled(crop_rect, im_temp, batch_img_size, croptype, crop_position);
					}
					else
						crop_patch(crop_rect, im_temp, scale_selected, croptype, crop_position);
				}
				crop_img(im_temp, im_temp, crop_rect, bFlipLR);
				crop_img(im_temp_lowres, im_temp_lowres, crop_rect, bFlipLR);
			}
			

			// 3. trans to float 
			processed_image.data.reset(new cv::Mat());
			im_temp.convertTo(*processed_image.data, CV_32F);

			im_temp_lowres.convertTo(im_temp_lowres, CV_32F);
			processed_image.additional_data["im_degraded"] = im_temp_lowres;

			// 4. resize im size to this->layer_param_.batchimgsize()
			//   4.2 resize
			//CHECK_EQ(processed_image.data->cols, batch_img_size) << "SuperResolution Crops are not allowed to resize";
			//CHECK_EQ(processed_image.data->rows, batch_img_size) << "SuperResolution Crops are not allowed to resize";
			if (croptype != data_utils::CropType_FullView) {
				if (processed_image.data->cols != batch_img_size || processed_image.data->rows != batch_img_size) {
					cv::resize(*processed_image.data, *processed_image.data, cv::Size(batch_img_size, batch_img_size), 0.0, 0.0, cv::INTER_CUBIC);
					LOG(FATAL) << "Warning: Resizing image!";
				}
			}

			// 7. minus mean 
			
			const float mean_value = entry_config.get<float>("mean_value");
			const float norm_value = entry_config.get<float>("norm_value");
			auto &im = *processed_image.data;
			auto &im1 = im_temp_lowres;
			int pixels = im.rows * im.cols * im.channels();
			float *pImg = (float*)im.ptr(0);
			float *pImg1 = (float*)im1.ptr(0);
			for (int x = 0; x < pixels; x++) {
				pImg[x] -= mean_value;
				pImg[x] /= norm_value;
				pImg1[x] -= mean_value;
				pImg1[x] /= norm_value;
			}
			return;
		}

		void image_sr_entry::degrade_one_image(data_container &processed_image) {
			caffepro_config_reader &provider_config = provider_.data_provider_config();
			caffepro_config_reader &entry_config = config_;

			auto &im = *processed_image.data;

			// crop with pad_inner
			const int pad_inner = entry_config.get<int>("pad_inner");
			cv::Rect cropRect = cv::Rect(pad_inner, pad_inner, im.cols - 2 * pad_inner, im.rows - 2 * pad_inner);
			im = im(cropRect).clone();

			return;
		}

		void image_sr_entry::addnoise_one_image(data_container &processed_image) {
			caffepro_config_reader &entry_config = config_;

			auto &im = *processed_image.data;	
			cv::Mat im_degraded;

			// add noise
			im_degraded = im.clone();
			const float noise_std = entry_config.get<float>("noise_std");
			int pixels = im.rows * im.cols * im.channels();
			float *pImg = (float*)im_degraded.ptr(0);
			for (int x = 0; x < pixels; x++)
				pImg[x] += (float)random_helper::normal_real() * noise_std;

			processed_image.additional_data["im_degraded"] = im_degraded;

			// crop with pad_inner
			const int pad_inner = entry_config.get<int>("pad_inner");
			cv::Rect cropRect = cv::Rect(pad_inner, pad_inner, im.cols - 2*pad_inner, im.rows - 2*pad_inner);
			im = im(cropRect).clone();

			return;
		}

		void image_sr_entry::crop_patch(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, float fCropRatio,
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

		void image_sr_entry::crop_patch_not_scaled(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, int crop_size,
			data_utils::crop_type croptype, __in_opt data_utils::crop_position cropposition) {


			int height = matRawImg.rows, width = matRawImg.cols;
			if (crop_size > std::min(height, width)) {
				//LOG(FATAL) << "CropImg::Invalid crop_size!";
				//LOG(ERROR) << "Warning: CropImg::Invalid crop_size!";
				crop_size = std::min(height, width);
			}

			int crop_size_x = crop_size, crop_size_y = crop_size; // default crop size
			int x_off = -1, y_off = -1;

			switch (croptype) {
			case data_utils::CropType_Random:
				y_off = random_helper::uniform_int() % (height - crop_size + 1);
				x_off = random_helper::uniform_int() % (width - crop_size + 1);
				break;

			default:
				LOG(FATAL) << "CropImg::Undefined Crop Type!" << std::endl;
				break;
			}

			cropRect = cv::Rect(x_off, y_off, crop_size_x, crop_size_y);
		}

		void image_sr_entry::crop_img(__out cv::Mat &matOutImg, __in const cv::Mat &matRawImg, __in cv::Rect &cropRect, bool bFlipLR) {
			matOutImg = matRawImg(cropRect).clone();

			if (bFlipLR) cv::flip(matOutImg, matOutImg, 1);
		}
	}
}