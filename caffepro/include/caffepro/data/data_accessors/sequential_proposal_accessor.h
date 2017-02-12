
#pragma once 
#include <caffepro/object_model/data_model/data_accessor.h>

namespace caffepro {
	namespace data_model {
		class sequential_proposal_accessor : public data_accessor {
		public:
			sequential_proposal_accessor(data_provider &provider);
			~sequential_proposal_accessor();

		public:
			virtual void init();
			virtual void get_batch(batch_descriptor &batch);

		public:
			// fetch functions
			const std::vector<int> &data_indexes() const { return data_indexes_; }

		protected:
			std::vector<int> data_indexes_;
			int batch_size_;
			int start_index_, start_proposal_, start_view_;
			bool random_shuffle_;
			int views_;
		};
	}
}