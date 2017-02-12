
#pragma once 

#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/utils/data_utils/bigfile.h>
#include <caffepro/utils/data_utils/color_kl.h>
#include <caffepro/utils/data_utils/img_crop.h>

namespace caffepro {
	class data_bigfile_layer : public caffepro_layer {
	public:

		// definations
		struct bigfile_database {
			std::vector<data_utils::raw_picture> imgs;
			cv::Mat data_mean;
			data_utils::kl_infos class_info;

			int multilabel_classes; // for multi_label dataset, it is the number of classes; otherwise, it will be -1
			std::map<std::string, std::vector<int> > clsname_to_multilabel, picname_to_multilabel; // for multi_label_data
		};

		struct batch_state {
			std::vector<data_utils::raw_picture> processing_imgs;
			int image_start_idx;
			int fixed_view;
		};

		struct batch_prefetch {
			boost::shared_ptr<node_blob> images;
			boost::shared_ptr<node_blob> labels;
			std::vector<boost::shared_ptr<node_blob> > extra_data;
			batch_state prefetch_batch_state;
		};
	
	public:
		data_bigfile_layer(caffepro_context *context, const LayerParameter &param);
		~data_bigfile_layer();

	public:
		// overrides
		virtual void init();
		virtual void resize();

	public:
		// fetch functions
		batch_state &current_batch() { return current_batch_state_; }
		const batch_state &current_batch() const { return current_batch_state_; }
		batch_prefetch &prefetch_state() { return prefetch_; }
		const batch_prefetch &prefetch_state() const { return prefetch_; }
		bigfile_database &database() { return database_; }

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_after_forward();

	protected:
		// init functions
		void init_classid(std::vector<std::string> &folder_pathes, int num_classes);
		void init_picture_database();
		void setup_extra_data_processer(int data_extra_num);

		// process images
		static void* data_prefetch(void* layer_pointer);
		void prepare_one_image(
			__out cv::Mat &im, 
			__in data_utils::raw_picture &raw_data, 
			__in int itemid, 
			__in int total_channels, 
			__in_opt const cv::Mat meanIm = cv::Mat(), 
			int view_id = -1
			);
		void write_image_to_blob(data_utils::crop_type croptype, const std::vector<cv::Mat> &images, boost::shared_ptr<node_blob> blob);

	public:
		// data relationed
		int view_num(data_utils::crop_type croptype);
		bool is_fixed_size(data_utils::crop_type croptype);
		void crop_patch(
			__out cv::Rect &cropRect, 
			__in const cv::Mat &matRawImg, 
			float fCropRatio, 
			data_utils::crop_type croptype, 
			__in_opt data_utils::crop_position cropposition = data_utils::CropPosition_Begin
			);
		void crop_patch_nonuniform(
			__out cv::Rect &cropRect,
			__in const cv::Mat &matRawImg,
			int target_w,
			int target_h,
			data_utils::crop_type croptype,
			__in_opt data_utils::crop_position cropposition);
		void crop_img(
			__out cv::Mat &matOutImg,
			__in const cv::Mat &matRawImg,
			__in cv::Rect &cropRect,
			bool bFlipLR);

	protected:
		bigfile_database database_;
		batch_prefetch prefetch_;
		batch_state current_batch_state_;

		std::vector<int> num_start_, num_size_; // for gpu split attribute
		std::string data_prefetch_event_name_;
	};
}