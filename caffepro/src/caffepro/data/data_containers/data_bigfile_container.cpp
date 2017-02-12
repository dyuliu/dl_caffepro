
#include <caffepro/data/data_containers/data_bigfile_container.h>
#include <fstream>

namespace caffepro {
	namespace data_model {
		data_container* data_bigfile_container::create_new() {
			return new data_bigfile_container();
		}

		data_container* data_bigfile_container::clone() {
			data_bigfile_container *result = dynamic_cast<data_bigfile_container *>(data_container::clone());
			CHECK(result);

			result->data_length = data_length;
			result->data_offset = data_offset;

			return result;
		}

		void data_bigfile_container::load_data() {
			data.reset(new cv::Mat(1, (int)data_length, CV_8U));
			std::ifstream in(storage_path, std::ios::binary);
			in.seekg(data_offset, std::ios::beg);
			in.read((char *)data->ptr(), data_length);
		}

		void data_bigfile_container::unload_data() {
			data.reset(new cv::Mat());
		}
	}
}