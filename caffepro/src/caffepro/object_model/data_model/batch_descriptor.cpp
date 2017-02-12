
#include <caffepro/object_model/data_model/batch_descriptor.h>

namespace caffepro {
	namespace data_model {
		batch_descriptor::batch_descriptor() {
			// nothing to do
		}

		batch_descriptor::~batch_descriptor() {
			// nothing to do
		}

		batch_descriptor* batch_descriptor::create_new() {
			return new batch_descriptor();
		}
	}
}