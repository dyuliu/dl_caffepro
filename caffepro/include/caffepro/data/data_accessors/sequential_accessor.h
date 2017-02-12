
#pragma once 
#include <caffepro/object_model/data_model/data_accessor.h>

namespace caffepro {
	namespace data_model {
		class sequential_accessor : public data_accessor {
		public:
			sequential_accessor(data_provider &provider);
			~sequential_accessor();

		public:
			virtual void init();
			virtual void get_batch(batch_descriptor &batch);
			int get_batch_size() { return batch_size_; };
		
		protected:
			std::vector<int> data_indexes_;
			int batch_size_;
			int start_index_;
			bool random_shuffle_;
			int iter_;
		};
	}
}