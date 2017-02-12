
#pragma once

#include "opencv2/opencv.hpp"
#include "opencv/cv.h"

#include <string>

namespace caffepro {
	namespace data_utils {
		class xml_image {
		public:
			static void load_image(__out cv::Mat &matMeanImg, __in const std::string &file);
			static void save_image(__in const cv::Mat &matMeanImg, __in std::string file);
		};
	}
}