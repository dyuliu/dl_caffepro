
#pragma once 

#include <caffepro/object_model/caffepro_object.h>

namespace caffepro {
	namespace data_model {
		class data_provider;

		class data_loader : public caffepro_object {
		public:
			data_loader(data_provider &provider) : provider_(provider) {}
			virtual ~data_loader() {}

		public:
			virtual void load() = 0;

		protected:
			data_provider &provider_;
		};
	}
}