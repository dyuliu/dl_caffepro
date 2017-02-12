
#pragma once 
#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>

namespace caffepro {
	namespace data_model {
		class data_provider;

		class data_accessor : public caffepro_object {
		public:
			data_accessor(data_provider &provider) : provider_(provider) {}
			virtual ~data_accessor() {}

		public:
			// fetch functions
			data_provider &provider() const { return provider_; }

		public:
			virtual void init() {}
			virtual void get_batch(batch_descriptor &batch) = 0;

		protected:
			data_provider &provider_;
		};
	}
}