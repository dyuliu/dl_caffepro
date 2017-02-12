
#include <caffepro/data/data_loaders/kl_data_loader.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/utils/data_utils/color_kl.h>
#include <caffepro/utils/filesystem.h>
#include <fstream>

namespace caffepro {
	namespace data_model {
		using std::string;
		using std::vector;
		using std::set;

		kl_data_loader::kl_data_loader(data_provider &provider)
			: data_loader(provider) {
			// nothing to do
		}

		kl_data_loader::~kl_data_loader() {
			// nothing to do
		}

		void kl_data_loader::load() {
			LOG(INFO) << "Loading KL info";

			const string color_kl_dir = provider_.data_provider_config().get<string>("color_kl_dir");
			set<string> classes;
			for (int i = 0; i < (int)provider_.dataset_size(); i++) {
				classes.insert(provider_.get_data(i)->class_name);
			}

			for (set<string>::iterator iter = classes.begin(); iter!= classes.end(); ++iter) {
				string path = color_kl_dir + "/" + *iter;

				std::ifstream in(path);

				string pic_name_buf;
				while (in >> pic_name_buf) {
					data_utils::kl_info kl;
					for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
							in >> kl.P[j][i]; // col first, so P[j][i]
						}
					}
					for (int i = 0; i < 3; i++) {
						in >> kl.SqrtV[i];

						if (kl.SqrtV[i] > 0) {
							kl.SqrtV[i] = sqrt(kl.SqrtV[i]);
						}
						else {
							kl.SqrtV[i] = 0;
						}
					}

					// skip the last 3 numbers
					// which stand for means
					float tmp;
					for (int i = 0; i < 3; i++) {
						in >> tmp;
					}

					// fill in the database
					boost::shared_ptr<data_container> data = provider_.get_data(pic_name_buf);
					if (data) {
						data->additional_data["kl_info"] = kl;
					}
				}
			}
		}
	}
}