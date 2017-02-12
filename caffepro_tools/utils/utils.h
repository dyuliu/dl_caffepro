#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffepro/proto/caffe.pb.h>
#include <caffepro/caffepro.h>
#include <caffepro/utils/filesystem.h>
#include <caffepro/utils/color_print.h>
#include <caffepro/object_model/device_blob.h>

#include <omp.h>

// string & num transfer
namespace caffepro {

	std::string cat_file_num(std::string fold_name, std::string file_name, int num = -1, std::string app = ".log");

	// save data layer
	void save_layer_data(std::string tgt_fold, boost::shared_ptr<device_blob> &blob, int num = -1);

	// general save data
	void save_vector_data(std::vector<data_type> &data, std::string filename);
	void open_vector_data(std::vector<data_type> &data, std::string filename);


	// big <-> imgs
	bool big2imgs_batch(std::string src);
	bool imgs2big_batch(std::string src);

	// get images mean
	bool imgs2mean(std::string src);

	// extract DUMP
	void extract_dump();
	void decode_by_layer(std::vector<std::vector<data_type>> &dumps_);

	void extract_model(std::string proto_file, std::string proto_bin, bool isfolder);
	void extract_grad(std::string proto_file, std::string proto_bin, bool isfolder);

	// gpu search
	void gpu_info(int id);

}

