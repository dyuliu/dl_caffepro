
#pragma once 

#include <caffepro/object_model/data_model/data_entry.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>
#include <caffepro/utils/data_utils/img_crop.h>
#include <opencv2/opencv.hpp>

namespace caffepro {
	namespace data_model {
		class image_sr_entry : public data_entry {
		public:
			image_sr_entry(data_provider &provider, const std::string &name);
			virtual ~image_sr_entry();

		public:
			virtual void init();
			virtual void prepare(batch_descriptor &batch);

		protected:
			void prepare_one_image(const data_container &original_image, data_container &processed_image);
			void degrade_one_image(data_container &processed_image);
			
			void addnoise_one_image(data_container &processed_image);

			void crop_patch(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, float fCropRatio,
				data_utils::crop_type croptype, __in_opt data_utils::crop_position cropposition);
			void crop_patch_not_scaled(__out cv::Rect &cropRect, __in const cv::Mat &matRawImg, int crop_size,
				data_utils::crop_type croptype, __in_opt data_utils::crop_position cropposition);
			void crop_img(__out cv::Mat &matOutImg, __in const cv::Mat &matRawImg, __in cv::Rect &cropRect, bool bFlipLR);

		protected:
			cv::Mat data_mean_;
		};
	}
}