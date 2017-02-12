
#pragma once 
#include <caffepro/object_model/data_model/data_accessor.h>

namespace caffepro {
	namespace data_model {
		class sequential_rank_accessor : public data_accessor {
		public:
			sequential_rank_accessor(data_provider &provider);
			~sequential_rank_accessor();

		public:
			virtual void init();
			virtual void get_batch(batch_descriptor &batch);
			void update_score(const std::string &data_name, data_type new_score);

		protected:
			std::vector<int> data_indexes_;
			int batch_size_;
			int ranking_batch_size_;
			int start_index_;
			bool random_shuffle_;
			std::map<std::string, data_type> scores_;
		};
	}
}