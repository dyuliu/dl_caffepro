
#include <caffepro/object_model/data_model/data_container.h>

namespace caffepro {
	namespace data_model {
		data_container::data_container() {
			// nothing to do
		}

		data_container::~data_container() {
			// nothing to do
		}

		data_container* data_container::create_new() {
			return new data_container();
		}

		data_container* data_container::clone() {
			data_container *result = create_new();

			// we only copy the reference of data - shallow copy
			result->data = data;

			result->data_name = data_name;
			result->storage_path = storage_path;
			result->class_name = class_name;
			result->label_id = label_id;

			result->width = width;
			result->height = height;
			result->raw_width = raw_width;
			result->raw_height = raw_height;
			result->label_ids = label_ids;
			result->groundtruth_boxes = groundtruth_boxes;
			result->proposal_boxes = proposal_boxes;
			result->additional_data = additional_data;

			return result;
		}
	}
}