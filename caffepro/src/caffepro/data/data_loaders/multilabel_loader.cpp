
#include <caffepro/data/data_loaders/multilabel_loader.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <fstream>

namespace caffepro {
	namespace data_model {
		using std::vector;
		using std::string;

		multilabel_loader::multilabel_loader(data_provider &provider) 
			: data_loader(provider) {
			// nothing to do
		}

		multilabel_loader::~multilabel_loader() {
			// nothing to do
		}

		void multilabel_loader::load() {
			LOG(INFO) << "Loading multi-label def file";
			caffepro_config_reader &config = provider_.data_provider_config();
			const string multilabel_file_name = config.get<string>("multilabel_def_file");

			std::ifstream stream(multilabel_file_name);
			int total = 0;

			string line;
			while (std::getline(stream, line)) {
				line = trim(line);
				if (line.empty()) continue;

				vector<string> sp = split(line, '\t');
				CHECK_GT(sp.size(), 1) << line;

				string elem_name = sp[0];
				vector<int> labels;
				for (int i = 1; i < (int)sp.size(); i++) {
					labels.push_back(string_to_int(sp[i]));
				}

				boost::shared_ptr<data_container> &img = provider_.get_data(elem_name);
				CHECK(img) << "Cannot find the image named " << elem_name;

				img->label_ids = labels;
				total++;
			}

			LOG(INFO) << total << " pictures have multi-label definations";
			
			// checking 
			LOG(INFO) << "Checking def file";
			for (int i = 0; i < (int)provider_.dataset_size(); i++) {
				boost::shared_ptr<data_container> &img = provider_.get_data(i);
				if (img->label_ids.size() == 0) {
					LOG(FATAL) << "Picture " << img->data_name << " in class " << img->class_name << " not defined in multi-label def file";
				}
			}
		}
	}
}