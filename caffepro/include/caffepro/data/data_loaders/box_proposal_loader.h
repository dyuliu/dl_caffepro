
#pragma once

#include <caffepro/object_model/data_model/data_loader.h>

namespace caffepro {
	namespace data_model {
		class box_proposal_loader : public data_loader {
		public:
			box_proposal_loader(data_provider &provider);
			virtual ~box_proposal_loader();

		public:
			virtual void load();
		};
	}
}