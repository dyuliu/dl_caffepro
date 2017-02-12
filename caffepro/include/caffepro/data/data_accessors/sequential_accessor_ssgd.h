
#pragma once 
#include <caffepro/object_model/data_model/data_accessor.h>
#include <caffepro/utils/multinode.h>

namespace caffepro {
	namespace data_model {
		class sequential_accessor_ssgd : public data_accessor {
		public:
			sequential_accessor_ssgd(data_provider &provider);
			~sequential_accessor_ssgd();

		public:
			virtual void init();
			virtual void get_batch(batch_descriptor &batch);
			void multinode_shffule();

		protected:
			std::vector<int> data_indexes_;
			int batch_size_;
			int start_index_;
			bool random_shuffle_;
		
		// ssgd
		protected:
			int worker_size_;
			int worker_id_;
			multinode* multinode_;
			data_type* worker_sequence_;
			int worker_start_idx_;
			int worker_end_idx_;
			int worker_data_size_;
			int iter_;
		};
	}
}