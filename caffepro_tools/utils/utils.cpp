
#include "utils.h"
#include <caffepro/utils/filesystem.h>
#include <fstream>

#include <caffepro/proto/proto_io.h>

#include <caffepro/updater/updaters.h>
#include <caffepro/solver/sgd_solver.h>

#include <caffepro/layers/data_bigfile_layer.h>

#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/object_model/caffepro_net.h>

#include <memory>
#include <cuda_runtime.h>
// #include <helper_cuda.h>
#include <cuda.h>

#define CUDA_CALL(x) { const cudaError_t cudaError_code = (x) ; \
	if ( cudaError_code != cudaSuccess) { \
		printf("\nCUDA Error: %s (err_num=%d)\n", cudaGetErrorString(cudaError_code), cudaError_code); \
		cudaDeviceReset(); assert(0); } }

namespace caffepro {
	/************************************************************************/
	// UTILS: GPU INFO
	// UPDATE_TIME: 2016-08-05
	/************************************************************************/
	void gpu_info(int id) {
		int count;
		CUDA_CALL(cudaGetDeviceCount(&count));
		cudaDeviceProp prop;
		std::ofstream fp("gpu_info_" + std::to_string(id) + ".txt");
		// for search
		for (int i = 0; i < count; i++) {
			CUDA_CALL(cudaGetDeviceProperties(&prop, i));
			fp << "---- General Information for device " << i << " ----" << std::endl;
			fp << "Name: " << prop.name << std::endl;
			fp << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
			fp << "Clock rate: " << prop.clockRate << std::endl;
			fp << "Total global mem: " << prop.totalGlobalMem << std::endl;
			fp << "Total constant mem: " << prop.totalConstMem << std::endl;
			fp << "Threads in warp: " << prop.warpSize << std::endl;
			fp << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
			fp << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
			fp << "Max share mem per block: " << prop.sharedMemPerBlock << std::endl;
		}
		fp.close();
	}

	/************************************************************************/
	// UTILS: STRING
	// UPDATE_TIME: 2016-08-05
	/************************************************************************/
	std::string cat_file_num(std::string fold_name, std::string file_name, int num, std::string app) {
		char temp[255];
		memset(temp, 0, 255);
		sprintf(temp, "%s//%s_%05d", fold_name.data(), file_name.data(), num);
		auto file_ = std::string(temp) + app;
		return file_;
	}

	void save_layer_data(std::string tgt_fold, boost::shared_ptr<device_blob> &blob, int num) {
		filesystem::create_directory(tgt_fold.data());
		std::ofstream fp(cat_file_num(tgt_fold, tgt_fold, num), std::ios::out);
		blob->save_data(fp);
		fp.close();
	}

	void save_vector_data(std::vector<data_type> &data, std::string filename) {
		// save vector length
		int size_ = (int)data.size();
		data_type *ptr = (data_type*)malloc(sizeof(data_type)*size_);
		for (int i = 0; i < size_; i++) ptr[i] = data[i];
		std::ofstream fp(filename, std::ios::out | std::ios::binary);
		fp.write((char*)&size_, sizeof(int));
		fp.write((char*)ptr, sizeof(data_type)*size_);
		free(ptr);
		fp.close();
	}

	void open_vector_data(std::vector<data_type> &data, std::string filename) {
		std::ifstream fp(filename, std::ios::in | std::ios::binary);
		int size_ = 0;
		fp.read((char*)&size_, sizeof(int));
		data_type *ptr = (data_type*)malloc(sizeof(data_type)*size_);
		fp.read((char*)ptr, sizeof(data_type)*size_);
		for (int i = 0; i < size_; i++) data.push_back(ptr[i]);
		fp.close();
	}

	/************************************************************************/
	// UTILS: IMAGE PROCESSING
	// UPDATE_TIME: 2016-08-05
	/************************************************************************/
	void save_image(__in const cv::Mat &matMeanImg, __in std::string file) {
		cv::FileStorage fs;
		fs.open(file, cv::FileStorage::WRITE);
		if (!fs.isOpened()) {
			std::cout << "xml_image :: Can't open " << file;
		}
		fs << "Channel" << matMeanImg.channels();
		fs << "Row" << matMeanImg.rows;
		fs << "Col" << matMeanImg.cols;
		cv::Mat mean = matMeanImg.clone();
		mean = mean.reshape(1, 1);
		fs << "MeanImg" << mean;
		fs.release();
	}

	bool imgs2mean(std::string src) {
		auto imgslist = caffepro::filesystem::get_files(src.data(), "*", true);
		auto img1 = cv::imread(imgslist[0]);
		int num = 0;
		int col = img1.cols, row = img1.rows;
		cv::Mat pg = cv::Mat::zeros(row, col, CV_32FC3);

		for (auto imgpath : imgslist) {
			if (num % 20 == 0) std::cout << "Processing: " << imgpath << " Finished: " << float(num) * 100 / imgslist.size() << std::endl;
			auto img = cv::imread(imgpath, 1);
			for (int i = 0; i < col; i++)
				for (int j = 0; j < row; j++)
					for (int k = 0; k < 3; k++)
						pg.at<cv::Vec3f>(i, j)[k] += (float)img.at<cv::Vec3b>(i, j)[k];
			num++;
		}

		for (int i = 0; i < col; i++)
			for (int j = 0; j < row; j++)
				for (int k = 0; k < 3; k++)
					pg.at<cv::Vec3f>(i, j)[k] /= (float)(num + (0.0001));

		save_image(pg, src + "_mean.xml");
		return true;
	}

	/************************************************************************/
	// UTILS: EXTRACT *.MODEL INFO (WEIGHT AND GRADIENT)
	// UPDATE_TIME: 2016-08-05
	/************************************************************************/
	void extract_dump() {
		COUT_METD << "Start to aggregate the data" << std::endl;
		std::vector<std::vector<data_type>> dumps_;
		auto infos = caffepro::filesystem::get_files("dump", "*.*", false);
		std::vector<data_type> tmp_;
		for (int i = 0; i < infos.size(); i++) {
			tmp_.clear();
			open_vector_data(tmp_, infos[i]);
			dumps_.push_back(tmp_);
		}
		decode_by_layer(dumps_);
		COUT_SUCC << "Success on processing the dump info" << std::endl;
		COUT_SUCC << "Please check fold of dump_by_layer" << std::endl;
	}

	void decode_by_layer(std::vector<std::vector<data_type>> &dumps_) {
		COUT_METD << "Start to decode the data into layer" << std::endl;
		std::string fold_ = "dump_by_layer";
		caffepro::filesystem::create_directory(fold_.data());
		int size_dumps_ = (int)dumps_.size();
		int size_nums_ = (int)dumps_[0].size();
		for (int i = 0; i < size_nums_; i += 4) {
			std::string layer_id = cat_file_num(fold_, "layer_", int(i / 4 + 1));
			std::ofstream fp(layer_id, std::ios::out);
			for (int j = 0; j < size_dumps_; j++) {
				fp << j << '\t' << dumps_[j][i] << '\t' << dumps_[j][i + 1] << '\t' << dumps_[j][i + 2] << '\t' << dumps_[j][i + 3] << std::endl;
			}
			fp.close();
		}
	}

	void extract_model_process(std::string proto_file, std::string proto_bin) {
		NetParameter net_param;
		proto_io(net_param).from_text_file(proto_file);

		caffepro_context context;

		cudaSetDevice(0);

		caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);

		NetParameter weights;
		proto_io(weights).from_binary_file(proto_bin);
		net->load_weights(weights);
		auto weight_info_ = net->weights_info();

		int size_ = 0;
		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			size_ += weight_info_.weights[i]->get(0)->count();
		}

		auto proto_name = filesystem::get_file_name_without_extension(proto_bin);
		std::ofstream fp(proto_name + "_model.log", std::ios::out);

		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			for (int j = 0; j < weight_info_.weights[i]->get(0)->count(); j++) {
				fp << weight.cpu_data()[j] << std::endl;
			}
		}

		fp.close();
	}

	void extract_model(std::string proto_file, std::string proto_bin, bool isfolder = false) {
		COUT_METD << "Start to extract model value" << std::endl;

		if (isfolder) {
			auto bin_list = filesystem::get_files(proto_bin.data(), "*.model", false);
			for (auto bin : bin_list) {
				COUT_RUNN << "Processing " << bin << std::endl;
				extract_model_process(proto_file, bin);
			}
		}
		else {
			COUT_RUNN << "Processing " << proto_bin << std::endl;
			extract_model_process(proto_file, proto_bin);
		}
	}

	void extract_grad_process(std::string proto_file, std::string proto_bin) {
		NetParameter net_param;
		proto_io(net_param).from_text_file(proto_file);

		caffepro_context context;

		cudaSetDevice(0);

		caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);
		NetParameter weights;
		proto_io(weights).from_binary_file(proto_bin);
		net->load_weights(weights);
		auto weight_info_ = net->weights_info();

		int size_ = 0;
		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			size_ += weight_info_.weights[i]->get(0)->count();
		}

		std::vector<caffepro::caffepro_object::output_metric> metrics;
		SolverParameter solverparam;
		solverparam.set_momentum(float(0.9));

		sgd_updater updater(&context, weight_info_, solverparam, metrics);
		net->forward(false);
		context.sync_all_devices();

		net->backward(false);
		context.sync_all_devices();

		auto proto_name = filesystem::get_file_name_without_extension(proto_bin);
		std::ofstream fp(proto_name + "_grad.log", std::ios::out);

		for (int i = 0; i < (int)weight_info_.weights.size(); i++) {
			device_blob &weight = *weight_info_.weights[i]->get(0);
			for (int j = 0; j < weight_info_.weights[i]->get(0)->count(); j++) {
				fp << weight.cpu_diff()[j] << std::endl;
			}
		}

		fp.close();
	}

	void extract_grad(std::string proto_file, std::string proto_bin, bool isfolder = false) {
		COUT_METD << "Start to extract gradient value" << std::endl;

		if (isfolder) {
			auto bin_list = filesystem::get_files(proto_bin.data(), "*.model", false);
			for (auto bin : bin_list) {
				COUT_RUNN << "Processing " << bin << std::endl;
				extract_grad_process(proto_file, bin);
			}
		}
		else {
			COUT_RUNN << "Processing " << proto_bin << std::endl;
			extract_grad_process(proto_file, proto_bin);
		}
	}



	/************************************************************************/
	// UTILS: BIGS <-> IMAGES
	// UPDATE_TIME: 2016-08-05
	/************************************************************************/

	// convert a bigfile into a folder of images
	bool big2imgs(std::string src) {
		COUT_READ << "The source path is : " << src << std::endl;

		auto path = caffepro::filesystem::get_directory_name(src);
		auto fname = caffepro::filesystem::get_file_name_without_extension(src);
		path += fname;

		if (caffepro::filesystem::create_directory(path.data()))
			COUT_SUCC << "Create directory : " << path << std::endl;

		size_t totalsize = 0, count = 0;

		std::ifstream in(src, std::ios::binary);
		char sub_file_name[300];

		while (in) {
			int n_file_name;

			if (!in.read((char *)&n_file_name, sizeof(int))) break;;
			in.read(sub_file_name, n_file_name);
			sub_file_name[n_file_name] = '\0';
			int content_length;

			in.read((char *)&content_length, sizeof(int));

			auto data = cv::Mat(1, content_length, CV_8U);
			in.read((char*)data.ptr(), content_length);

			auto savedata = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
			cv::imwrite(path + "\\" + sub_file_name, savedata);

			totalsize += content_length;
			count++;
		}

		in.close();

		COUT_CHEK << "Total Size is " << totalsize << std::endl;
		COUT_CHEK << "Total Number is " << count << std::endl << std::endl;

		return true;
	}

	// convert a series of bigfiles into multi-folders of images
	// generate 2-layer folder
	bool big2imgs_batch(std::string src) {
		COUT_METD << "bigfile to images" << std::endl;
		auto biglist = caffepro::filesystem::get_files(src.data(), "*", false);
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < biglist.size(); i++)
		{
			auto bigfile = biglist[i];
			COUT_RUNN << "Thread " << omp_get_thread_num() << " and Finished: " << i*100.0 / biglist.size();
			big2imgs(bigfile);
		}
		return true;
	}

	// convert a folder of images into a bigfile
	bool imgs2big(std::string src) {
		COUT_READ << "The source path is : " << src << std::endl;

		std::vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(0);

		auto imgs = caffepro::filesystem::get_files(src.data(), "*.*", false);
		std::ofstream out((src + ".big").data(), std::ios::binary | std::ios::out);
		std::ofstream out_log((src + ".log").data(), std::ios::out);

		int count = 0;
		for (auto filename : imgs) {
			if (count++ == 20) {
				COUT_RUNN << filename << std::endl;
				count = 0;
			}
			out_log << filename << std::endl;
			auto shortname = caffepro::filesystem::get_file_name(filename);
			int n_file_name = (int)strlen(shortname.data());
			// write length of filename
			out.write((char*)&n_file_name, sizeof(int));
			// wirte filename
			out.write(shortname.data(), n_file_name);
			// encode img
			auto img = cv::imread(filename);
			std::vector<uchar> buf;
			cv::imencode(".PNG", img, buf, compression_params);
			int f_size = (int)buf.size();
			// write imgs length
			out.write((char*)&f_size, sizeof(int));
			// write imgs
			for (auto wd : buf) out.write((char*)&wd, sizeof(uchar));
		}
		out.close();
		out_log.close();
		return true;
	}

	// convert a series of folders of images into a bigfiles
	// this is 2-layer folder
	bool imgs2big_batch(std::string src) {
		COUT_METD << "Images to big file" << std::endl;
		auto imgslist = caffepro::filesystem::get_directories(src.data(), "*", false);
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < imgslist.size(); i++)
		{
			auto imgs = imgslist[i];
			COUT_RUNN << "Thread " << omp_get_thread_num() << " and Finished: " << i*100.0 / imgslist.size() << std::endl;
			imgs2big(imgs);
		}
		return true;
	}


}