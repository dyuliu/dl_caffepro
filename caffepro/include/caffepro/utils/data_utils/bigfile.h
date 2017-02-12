
#pragma once

#include <caffepro/utils/data_utils/box.h>

#include <string>
#include <vector>
#include <random>

#include "opencv2/opencv.hpp"
#include "opencv/cv.h"

namespace caffepro {
	namespace data_utils {

		// picture descriptor
		class raw_picture {
		public:
			// required attributes
			std::string			path;							// picture name
			std::string			big_file_path;					// big file path
			int					label_id;						// label id
			size_t				data_length;					// data length in big file
			long long			data_offset;					// data offset in big file
			cv::Mat				data;							// data content, in binary format (may be empty, for !cache_data) (never flipped)

			// for view training
			std::vector<int> views;
			int cur_view;

			// optional attributes (available when metadata provided) 
			int					width;							// image width
			int					height;							// image height
			int					raw_width;						// raw image width
			int					raw_height;						// raw image height
			std::vector<Box>	bounding_boxes;					// bounding boxes (may be flipped)
			bool				flipped;						// whether flipped
			Box					crop_box;						// current crop box which is applied on the picture (may be flipped)
			std::vector<Box>	prop_boxes;						// proposal boxes (never flipped)

			// methods
		public:
			raw_picture();
			raw_picture(int nView);
			raw_picture(const std::string& s_path, int label, int nView);

			int get_current_view_idx();
			void load_data();
			void clear_data();
		};

		// big file related utils 
		size_t load_bigfile(__out std::vector<raw_picture> &container, __in int label_ids,
			const std::string& filename, int nView, bool load_data);

		// fill metadata if imgs is not empty, or push new image metadata if imgs is empty
		void load_metadata(const std::string &metadata_filename, const std::string &bigfile_folder, int nView,
			__in const std::map<std::string, int> &dic_classname2id, __inout std::vector<raw_picture> &imgs);

		void load_prop_boxes(const std::string &prop_filename, __inout std::vector<raw_picture> &imgs);

		void rawpicture_to_im(__out cv::Mat &im, __in const raw_picture & rawIm);
		void rawpicture_to_im_adapt(__out cv::Mat &im, __in const raw_picture & rawIm);
	}
}