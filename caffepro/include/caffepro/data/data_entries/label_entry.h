
#pragma once 

#include <caffepro/object_model/data_model/data_entry.h>

namespace caffepro {
	namespace data_model {
		class label_entry : public data_entry {
		public:
			label_entry(data_provider &provider, const std::string &name);
			~label_entry();

		public:
			virtual void init();
			virtual void prepare(batch_descriptor &batch);
		};
	}
}