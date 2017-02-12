
#pragma once

#include <caffepro/object_model/data_model/data_loader.h>

namespace caffepro {
	namespace data_model {
		class multilabel_loader : public data_loader {
		public:
			multilabel_loader(data_provider &provider);
			virtual ~multilabel_loader();

		public:
			virtual void load();
		};
	}
}