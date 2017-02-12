
#include <caffepro/data/data_entries/label_entry.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>
#include <caffepro/object_model/data_model/data_provider.h>

namespace caffepro {
	namespace data_model {
		label_entry::label_entry(data_provider &provider, const std::string &name)
			: data_entry(provider, name) {
			type_ = "label";
		}

		label_entry::~label_entry() {
			// nothing to do
		}

		void label_entry::init() {
			data_entry::init();
			
			const int batchsize = provider_.data_provider_config().get<int>("batch_size");
			auto_init_buffer(0, batchsize, 1, 1, 1, true);
		}

		void label_entry::prepare(batch_descriptor &batch) {
			int batchsize = (int)batch.batch_data.size();

			if (batchsize != prefetch_buffer_[0]->get(0)->count()) {
				prefetch_buffer_[0]->get(0)->reshape_4d(batchsize, 1, 1, 1);
			}

			data_type *label_data = prefetch_buffer_[0]->get(0)->write_only_cpu_data();
			for (int i = 0; i < batchsize; i++) {
				label_data[i] = (data_type)batch.batch_data[i].original_data->label_id;
			}
 		}
	}
}