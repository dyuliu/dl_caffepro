
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/utils/data_utils/box_trans.h>

namespace caffepro {

	class sigmoid_layer;

	class anchor_loc_loss_ex_layer : public caffepro_layer {
	public:
		anchor_loc_loss_ex_layer(caffepro_context *context, const LayerParameter &param);
		~anchor_loc_loss_ex_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	public:
		data_utils::box_anchor_transform get_transformer(const data_utils::Box &base_box, int im_width, int im_height);

	public:
		// fetch functions
		int num_anchors() const {
			return anchor_num_width_ * anchor_num_height_ * (int)central_scales_.size() * (int)aspect_ratio_.size();
		}

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);

	private:
		// sigmoid term
		boost::shared_ptr<sigmoid_layer> sigmoid_layer_;
		boost::shared_ptr<node_blob> sigmoid_output_;
		layer_io_buffer sigmoid_bottom_vec_, sigmoid_top_vec_;

		// anchor related
		int anchor_num_width_, anchor_num_height_;
		data_type spatial_start_, spatial_step_;
		std::vector<data_type> central_scales_, aspect_ratio_;

		// layer params
		data_type loc_threshold_;
		int total_classes_;

		bool auto_spatial_anchor_;
		data_type loc_coeff_, conf_coeff_, assign_reject_iou_, cls_pos_iou_, cls_neg_iou_;
		bool prediction_box_classification_;
		std::vector<data_type> correct_;

		// buffers
		boost::shared_ptr<node_blob> loc_pos_entry_, conf_pos_entry_;
		boost::shared_ptr<node_blob> loc_diff_buf_, conf_diff_buf_, loc_data_buf_, conf_data_buf_, conf_raw_data_buf_;

		// optional params
		int expected_pos_num_, expected_neg_num_;
	};
}