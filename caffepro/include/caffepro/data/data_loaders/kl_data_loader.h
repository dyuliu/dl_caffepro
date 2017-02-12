
#pragma once 

#include <caffepro/object_model/data_model/data_loader.h>

namespace caffepro {
	namespace data_model {
		class kl_data_loader : public data_loader {
		public:
			kl_data_loader(data_provider &provider);
			virtual ~kl_data_loader();

		public:
			virtual void load();
		};
	}
}