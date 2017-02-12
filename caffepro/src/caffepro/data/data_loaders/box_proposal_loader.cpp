
#include <caffepro/data/data_loaders/box_proposal_loader.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/utils/data_utils/binary_io.h>
#include <caffepro/utils/data_utils/box.h>
#include <map>

namespace caffepro {
	namespace data_model {
		using std::string;
		using data_utils::Box;

		box_proposal_loader::box_proposal_loader(data_provider &provider)
			: data_loader(provider){
			// nothing to do
		}

		box_proposal_loader::~box_proposal_loader() {
			// nothing to do
		}

		void box_proposal_loader::load() {
			LOG(INFO) << "Loading box proposals";

			caffepro_config_reader &config = provider_.data_provider_config();
			const string prop_filename = config.get<string>("crop_box_file");

			std::ifstream in(prop_filename, std::ios::binary);
			data_utils::binary_reader reader(in);

			while (true) {
				string picname = reader.read<string>();
				if (!in) break;

				boost::shared_ptr<data_container> img = provider_.get_data(picname);
				CHECK(img);

				int nboxes = reader.read<int>();
				for (int i = 0; i < nboxes; i++) {
					int left = reader.read<int>();
					int top = reader.read<int>();
					int right = reader.read<int>();
					int bottom = reader.read<int>();
					int label = reader.read<int>();;
					Box box(left, top, right, bottom, label);
					img->proposal_boxes.push_back(box);

					reader.read<float>(); // skip confidence
				}
			}

			// display box loading information
			int nobox = 0, invalid_box = 0, box_outofrange = 0;
			for (int i = 0; i < (int)provider_.dataset_size(); i++) {
				if (provider_.get_data(i)->proposal_boxes.size() == 0) {
					nobox++;
				}
				else {
					int img_width = provider_.get_data(i)->width;
					int img_height = provider_.get_data(i)->height;

					for (int j = 0; j < (int)provider_.get_data(i)->proposal_boxes.size(); j++) {
						Box box = provider_.get_data(i)->proposal_boxes[j];
						if (!box.valid()) {
							invalid_box++;
						}
						else if (box.left < 0 || box.top < 0 || box.right >= img_width || box.bottom >= img_height) {
							box_outofrange++;
							Box image_box(0, 0, img_width - 1, img_height - 1);
							provider_.get_data(i)->proposal_boxes[j] = box.intersect(image_box);
						}
					}
				}
			}

			LOG(ERROR) << "Pictures without boxes: " << nobox;
			LOG(ERROR) << "Invalid boxes: " << invalid_box;
			LOG(ERROR) << "Boxes out of range: " << box_outofrange;
		}
	}
}