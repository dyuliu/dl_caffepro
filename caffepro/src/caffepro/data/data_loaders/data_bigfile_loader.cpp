
#include <caffepro/data/data_loaders/data_bigfile_loader.h>
#include <caffepro/data/data_containers/data_bigfile_container.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/utils/filesystem.h>
#include <caffepro/utils/data_utils/binary_io.h>
#include <caffepro/utils/data_utils/box.h>
#include <boost/scoped_ptr.hpp>
#include <fstream>

namespace caffepro {
	namespace data_model {
		using std::string;
		using std::vector;
		using data_utils::Box;

		data_bigfile_loader::data_bigfile_loader(data_provider &provider) 
			: data_loader(provider) {
			boost::scoped_ptr<data_container> container_temp(provider_.create_data_container());
			CHECK(dynamic_cast<data_bigfile_container *>(container_temp.get()))
				<< "data_bigfile_container is needed for data_bigfile_loader";
		}

		data_bigfile_loader::~data_bigfile_loader() {
			// nothing to do
		}
		
		void data_bigfile_loader::load() {
			LOG(INFO) << "Loading big-file dataset";

			const string data_source_folder = provider_.data_provider_config().get<string>("source");

			vector<string> vFileList;
			vFileList = filesystem::get_files(data_source_folder.c_str(), "*.big", false);

			if (vFileList.empty()) {
				LOG(FATAL) << "Big File Folder " << data_source_folder << " is missing or empty";
			}

			for (int i = 0; i < (int)vFileList.size(); i++) {
				string labelname = filesystem::get_file_name_without_extension(vFileList[i]);
				label2classID_[labelname] = i;
				classID2label_[i] = labelname;
			}
			
			caffepro_config_reader &config = provider_.data_provider_config();
			const string metadata_file = config.get<string>("metadata_file", false, "");
			if (!metadata_file.empty()) {
				load_metadata(metadata_file);
			}
			else {
				for (int i = 0; i < (int)vFileList.size(); i++) {
					LOG(INFO) << "Loading " << classID2label_[i] << " (" << i << ")";
					google::FlushLogFiles(0);
					load_bigfile(vFileList[i], i);
				}
			}

			bool cache_data = config.get<bool>("cache_data", false, true);
			if (cache_data) {
				LOG(INFO) << "Loading cache data";
				for (int i = 0; i < (int)provider_.dataset_size(); i++) {
					provider_.get_data(i)->load_data();
				}
			}

		}

		void data_bigfile_loader::load_metadata(const string &metadata_file) {
			const string data_source_folder = provider_.data_provider_config().get<string>("source");

			std::ifstream in(metadata_file, std::ios::binary);
			data_utils::binary_reader reader(in);

			int picname_maxlen = reader.read<int>();
			int clsname_maxlen = reader.read<int>();

			int processed_pics = 0;
			while (true) {
				string picname = reader.read_fixedlen_string(picname_maxlen);
				if (!in) break;

				processed_pics++;

				string clsname = reader.read_fixedlen_string(clsname_maxlen);
				CHECK(label2classID_.count(clsname)) << "Unknown class name in metadata file: " << clsname;

				int clsid = label2classID_.find(clsname)->second;

				int width = reader.read<int>();
				int height = reader.read<int>();
				int raw_width = reader.read<int>();
				int raw_height = reader.read<int>();
				int bigfile_contentlen = reader.read<int>();
				long long bigfile_contentoffset = reader.read<long long>();

				int nboxes = reader.read<int>();
				vector<Box> boxes(nboxes);
				for (int i = 0; i < nboxes; i++) {
					string box_clsname = reader.read_fixedlen_string(clsname_maxlen);

					int left = reader.read<int>();
					int top = reader.read<int>();
					int right = reader.read<int>();
					int bottom = reader.read<int>();

					if (box_clsname == clsname) {
						boxes[i] = Box(left, top, right, bottom, clsid); // since box_clsname == clsname, clsid == box_clsid
					}
					else {
						boxes[i] = Box(left, top, right, bottom, atoi(box_clsname.c_str()));
					}
				}

				// add new
				boost::shared_ptr<data_container> container(provider_.create_data_container());
				container->data_name = picname;
				container->storage_path = data_source_folder + "\\" + clsname + ".big";
				container->class_name = clsname;
				container->label_id = clsid;

				data_bigfile_container *big_file_container = dynamic_cast<data_bigfile_container *>(container.get());
				big_file_container->data_length = bigfile_contentlen;
				big_file_container->data_offset = bigfile_contentoffset;

				container->width = width;
				container->height = height;
				container->raw_width = raw_width;
				container->raw_height = raw_height;
				container->groundtruth_boxes = boxes;

				provider_.add_data(container);
			}

			LOG(INFO) << "Metadata of " << processed_pics << " pictures loaded";
		}

		void data_bigfile_loader::load_bigfile(const string &bigfile_path, int label_id) {
			const string data_source_folder = provider_.data_provider_config().get<string>("source");

			std::ifstream in(bigfile_path, std::ios::binary);
			char sub_file_name[300];

			while (in) {
				int n_file_name;

				if (!in.read((char *)&n_file_name, sizeof(int))) break;
				in.read(sub_file_name, n_file_name);
				sub_file_name[n_file_name] = '\0';
				int content_length;
				in.read((char *)&content_length, sizeof(int));

				boost::shared_ptr<data_container> container(provider_.create_data_container());
				container->data_name = sub_file_name;
				container->storage_path = bigfile_path;
				container->class_name = classID2label_[label_id];
				container->label_id = label_id;

				data_bigfile_container *big_file_container = dynamic_cast<data_bigfile_container *>(container.get());
				big_file_container->data_length = content_length;
				big_file_container->data_offset = in.tellg();

				provider_.add_data(container);

				// skip data
				in.seekg(content_length, std::ios::cur);
			}
		}
	}
}