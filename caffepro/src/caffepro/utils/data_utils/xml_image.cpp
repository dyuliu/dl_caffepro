
#pragma once

#include <caffepro/utils/data_utils/xml_image.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	namespace data_utils {
		void xml_image::load_image(__out cv::Mat &matMeanImg, __in const std::string &file) {
			cv::Mat matMeanImgTemp;
			cv::FileStorage fs;
			fs.open(file, cv::FileStorage::READ);
			if (!fs.isOpened()) {
				LOG(FATAL) << "xml_image :: Can't open " << file;
			}
			fs["MeanImg"] >> matMeanImgTemp;
			int channel, row, col;
			fs["Channel"] >> channel;
			fs["Row"] >> row;
			fs["Col"] >> col;
			if (channel * row * col != matMeanImgTemp.channels() * matMeanImgTemp.rows * matMeanImgTemp.cols) {
				LOG(FATAL) << "xml_image : " << file << " broken!";
			}
			fs.release();
			matMeanImg = matMeanImgTemp.reshape(channel, row);
		}

		void xml_image::save_image(__in const cv::Mat &matMeanImg, __in std::string file) {
			cv::FileStorage fs;
			fs.open(file, cv::FileStorage::WRITE);
			if (!fs.isOpened()) {
				LOG(FATAL) << "xml_image :: Can't open " << file;
			}
			fs << "Channel" << matMeanImg.channels();
			fs << "Row" << matMeanImg.rows;
			fs << "Col" << matMeanImg.cols;
			cv::Mat mean = matMeanImg.clone();
			mean = mean.reshape(1, 1);
			fs << "MeanImg" << mean;
			fs.release();
		}
	}
}