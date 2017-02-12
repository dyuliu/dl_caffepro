
#include <caffepro/utils/data_utils/box_trans.h>
#include <caffepro/object_model/caffepro_config.h>

namespace caffepro {
	namespace data_utils {

		template <typename DTYPE>
		inline DTYPE cootrans(DTYPE value, DTYPE base, DTYPE unit) {
			return (value - base) / unit;
		}

		template <typename DTYPE>
		inline DTYPE cootrans_back(DTYPE value, DTYPE base, DTYPE unit) {
			return value * unit + base;
		}

		box_anchor_transform::box_anchor_transform(int im_width, int im_height, int num_width_anchors, int num_height_anchors, 
			const Box &crop_box, const std::vector<data_type> &central_scales, const std::vector<data_type> &aspect_ratio, float spatial_start, float spatial_step)
			: im_width_(im_width), im_height_(im_height),
			num_width_anchors_(num_width_anchors), num_height_anchors_(num_height_anchors),
			num_scale_anchors_((int)central_scales.size()), num_ratio_anchors_((int)aspect_ratio.size()) {
			CHECK_GE(num_width_anchors, 1);
			CHECK_GE(num_height_anchors, 1);
			CHECK(crop_box.valid());

			for (int r = 0; r < num_ratio_anchors_; r++) {
				for (int s = 0; s < num_scale_anchors_; s++) {
					for (int h = 0; h < num_height_anchors; h++) {
						for (int w = 0; w < num_width_anchors; w++) {
							data_type grid_center_x = (data_type)crop_box.left + (w * spatial_step + spatial_start) * crop_box.width();
							data_type grid_center_y = (data_type)crop_box.top + (h * spatial_step + spatial_start) * crop_box.height();
							data_type half_scale_x = crop_box.width() * central_scales[s] / 2;
							data_type half_scale_y = crop_box.height() * central_scales[s] / 2;

							data_type ratio = aspect_ratio[r];
							if (ratio != 1) {
								half_scale_y = sqrt(half_scale_x * half_scale_y / ratio);
								half_scale_x = ratio * half_scale_y;
							}

							anchors_.push_back(Box(
								(int)(grid_center_x - half_scale_x),
								(int)(grid_center_y - half_scale_y),
								(int)(grid_center_x + half_scale_x - 1),
								(int)(grid_center_y + half_scale_y - 1),
								-1
								));
						}
					}
				}
			}
		}

		box_anchor_transform::box_anchor_transform(int im_width, int im_height, const Box &anchor_box)
			: im_width_(im_width), im_height_(im_height),
			num_width_anchors_(1), num_height_anchors_(1),
			num_scale_anchors_(1), num_ratio_anchors_(1) {

			CHECK(anchor_box.valid());
			anchors_.push_back(anchor_box);
		}

		box_anchor_transform::box_anchor_transform(const Box &crop_box, int im_width, int im_height, const caffepro_config_reader &config) 
			: im_width_(im_width), im_height_(im_height) {
			
			num_width_anchors_ = config.get<int>("spatial_width");
			num_height_anchors_ = config.get<int>("spatial_height");

			std::vector<data_type> central_scales = config.get_array<data_type>("central_scale", false);
			if (central_scales.empty()) central_scales.push_back((data_type)1.f);
			std::vector<data_type> aspect_ratio = config.get_array<data_type>("aspect_ratio", false);
			if (aspect_ratio.empty()) aspect_ratio.push_back((data_type)1.f);

			num_scale_anchors_ = (int)central_scales.size();
			num_ratio_anchors_ = (int)aspect_ratio.size();

			data_type spatial_start = config.get<data_type>("spatial_start");
			data_type spatial_step = config.get<data_type>("spatial_step");

			CHECK_GE(num_width_anchors_, 1);
			CHECK_GE(num_height_anchors_, 1);
			CHECK(crop_box.valid());

			for (int r = 0; r < num_ratio_anchors_; r++) {
				for (int s = 0; s < num_scale_anchors_; s++) {
					for (int h = 0; h < num_height_anchors_; h++) {
						for (int w = 0; w < num_width_anchors_; w++) {
							data_type grid_center_x = (data_type)crop_box.left + (w * spatial_step + spatial_start) * crop_box.width();
							data_type grid_center_y = (data_type)crop_box.top + (h * spatial_step + spatial_start) * crop_box.height();
							data_type half_scale_x = crop_box.width() * central_scales[s] / 2;
							data_type half_scale_y = crop_box.height() * central_scales[s] / 2;

							data_type ratio = aspect_ratio[r];
							if (ratio != 1) {
								half_scale_y = sqrt(half_scale_x * half_scale_y / ratio);
								half_scale_x = ratio * half_scale_y;
							}

							anchors_.push_back(Box(
								(int)(grid_center_x - half_scale_x),
								(int)(grid_center_y - half_scale_y),
								(int)(grid_center_x + half_scale_x - 1),
								(int)(grid_center_y + half_scale_y - 1),
								-1
								));
						}
					}
				}
			}
		}

		box_t<data_type> box_anchor_transform::transform(const Box &box, 
			int anchor_width_index, int anchor_height_index, int anchor_scale_index, int anchor_ratio_index) {
			CHECK_LT(anchor_height_index, num_height_anchors_);
			CHECK_LT(anchor_width_index, num_width_anchors_);
			CHECK_LT(anchor_scale_index, num_scale_anchors_);
			CHECK_LT(anchor_ratio_index, num_ratio_anchors_);

			int anchor_index = ((anchor_ratio_index * num_scale_anchors_ + anchor_scale_index) * num_height_anchors_ + anchor_height_index) * num_width_anchors_ + anchor_width_index;
			return transform(box, anchor_index);
		}

		box_t<data_type> box_anchor_transform::transform(const Box &box, int anchor_index) {
			CHECK_LT(anchor_index, anchors_.size());

			Box anchor_box = anchors_[anchor_index];

			data_type x_unit = (data_type)anchor_box.width() / 2;
			data_type y_unit = (data_type)anchor_box.height() / 2;
			data_type center_x = (data_type)(anchor_box.left + anchor_box.right) / 2;
			data_type center_y = (data_type)(anchor_box.top + anchor_box.bottom) / 2;

			return box_t<data_type>(
				cootrans((data_type)box.left, center_x, x_unit),
				cootrans((data_type)box.top, center_y, y_unit),
				cootrans((data_type)box.right, center_x, x_unit),
				cootrans((data_type)box.bottom, center_y, y_unit),
				box.label_id
				);

			return box_t<data_type>::get_invalid_box();
		}

		Box box_anchor_transform::transform_back(const box_t<data_type> &box_trans,
			int anchor_width_index, int anchor_height_index, int anchor_scale_index, int anchor_ratio_index, bool fix_border) {
			CHECK_LT(anchor_height_index, num_height_anchors_);
			CHECK_LT(anchor_width_index, num_width_anchors_);
			CHECK_LT(anchor_scale_index, num_scale_anchors_);
			CHECK_LT(anchor_ratio_index, num_ratio_anchors_);

			int anchor_index = ((anchor_ratio_index * num_scale_anchors_ + anchor_scale_index) * num_height_anchors_ + anchor_height_index) * num_width_anchors_ + anchor_width_index;
			return transform_back(box_trans, anchor_index, fix_border);
		}

		Box box_anchor_transform::transform_back(const box_t<data_type> &box_trans, int anchor_index, bool fix_border) {
			CHECK_LT(anchor_index, anchors_.size());

			Box anchor_box = anchors_[anchor_index];

			data_type x_unit = (data_type)anchor_box.width() / 2;
			data_type y_unit = (data_type)anchor_box.height() / 2;
			data_type center_x = (data_type)(anchor_box.left + anchor_box.right) / 2;
			data_type center_y = (data_type)(anchor_box.top + anchor_box.bottom) / 2;

			Box result(
				(int)(cootrans_back(box_trans.left, center_x, x_unit) + 0.5),
				(int)(cootrans_back(box_trans.top, center_y, y_unit) + 0.5),
				(int)(cootrans_back(box_trans.right, center_x, x_unit) + 0.5),
				(int)(cootrans_back(box_trans.bottom, center_y, y_unit) + 0.5),
				box_trans.label_id
				);

			if (fix_border) {
				Box im_box(0, 0, im_width_ - 1, im_height_ - 1, -1);
				result = result.intersect(im_box);
			}

			return result;
		}

		Box box_anchor_transform::get_maxiou_box(int anchor_width_index, int anchor_height_index, int anchor_scale_index, int anchor_ratio_index, 
			const std::vector<Box> &boxes, __out data_type *corres_iou) {
			CHECK_LT(anchor_height_index, num_height_anchors_);
			CHECK_LT(anchor_width_index, num_width_anchors_);
			CHECK_LT(anchor_scale_index, num_scale_anchors_);
			CHECK_LT(anchor_ratio_index, num_ratio_anchors_);

			int anchor_index = ((anchor_ratio_index * num_scale_anchors_ + anchor_scale_index) * num_height_anchors_ + anchor_height_index) * num_width_anchors_ + anchor_width_index;
			return get_maxiou_box(anchor_index, boxes, corres_iou);
		}

		Box box_anchor_transform::get_maxiou_box(int anchor_index, const std::vector<Box> &boxes, __out data_type *corres_iou) {
			CHECK_GT(boxes.size(), 0);
			CHECK_LT(anchor_index, anchors_.size());

			int index = -1;
			float max_iou = -FLT_MAX;
			Box anchor_box = anchors_[anchor_index];

			for (int i = 0; i < (int)boxes.size(); i++) {
				float iou = (float)anchor_box.IoU(boxes[i]);
				if (iou > max_iou) {
					max_iou = iou;
					index = i;
				}
			}

			if (corres_iou) {
				*corres_iou = max_iou;
			}

			return boxes[index];
		}
	}
}