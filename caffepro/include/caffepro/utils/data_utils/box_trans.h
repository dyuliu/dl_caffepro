
#pragma once

#include <caffepro/utils/data_utils/box.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	class caffepro_config_reader;

	namespace data_utils {

		class box_anchor_transform {
		public:
			box_anchor_transform() {}
			box_anchor_transform(int im_width, int im_height, int num_width_anchors, int num_height_anchors, const Box &crop_box, const std::vector<data_type> &central_scales, const std::vector<data_type> &aspect_ratio, float spatial_start, float spatial_step);
			box_anchor_transform(int im_width, int im_height, const Box &anchor_box); // case for only one anchor box
			box_anchor_transform(const Box &crop_box, int im_width, int im_height, const caffepro_config_reader &config);

			box_t<data_type> transform(const Box &box, int anchor_width_index, int anchor_height_index, int anchor_scale_index, int anchor_ratio_index);
			box_t<data_type> transform(const Box &box, int anchor_index);
			Box transform_back(const box_t<data_type> &box_trans, int anchor_width_index, int anchor_height_index, int anchor_scale_index, int anchor_ratio_index, bool fix_border = true);
			Box transform_back(const box_t<data_type> &box_trans, int anchor_index, bool fix_border = true);
			Box get_maxiou_box(int anchor_width_index, int anchor_height_index, int anchor_scale_index, int anchor_ratio_index, const std::vector<Box> &boxes, __out data_type *corres_iou = NULL);
			Box get_maxiou_box(int anchor_index, const std::vector<Box> &boxes, __out data_type *corres_iou = NULL);

			std::vector<Box>& anchors() { return anchors_; }

		private:
			int im_width_, im_height_;
			int num_width_anchors_, num_height_anchors_, num_scale_anchors_, num_ratio_anchors_;

			std::vector<Box> anchors_; // (w, h, s)
		};
	}
}