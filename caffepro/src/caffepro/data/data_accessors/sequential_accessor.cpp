
#include <caffepro/data/data_accessors/sequential_accessor.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <random>
#include <algorithm>

#include <string>
#include <fstream>

namespace caffepro {
	namespace data_model {
		sequential_accessor::sequential_accessor(data_provider &provider)
			: data_accessor(provider), batch_size_(0), start_index_(0), random_shuffle_(false), iter_(0) {
			// nothing to do
		}

		sequential_accessor::~sequential_accessor() {
			// nothing to do
		}

		void print_sequence(int iter_, std::vector<int> values) {
			auto filename = std::to_string(iter_);
			std::ofstream fp(filename, std::ios::out);
			for (auto val : values)
				fp << val << std::endl;
			fp.close();
		}

		void sequential_accessor::init() {
			data_accessor::init();
			int data_size = (int)provider_.dataset_size();
			data_indexes_.resize(data_size);
			for (int i = 0; i < data_size; i++) {
				data_indexes_[i] = i;
			}
			caffepro_config_reader &config = provider_.data_provider_config();
			batch_size_ = config.get<int>("batch_size");
			random_shuffle_ = config.get<bool>("random_shuffle", false, false);

			if (random_shuffle_) {
				srand(std::random_device()());
				std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
			}
			else {
				srand(iter_);
				std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
			}
		}

		void sequential_accessor::get_batch(batch_descriptor &batch) {
			batch.batch_data.resize(batch_size_);
			for (int itemid = 0; itemid < batch_size_; itemid++) {
				int idx = data_indexes_[(start_index_ + itemid) % (int)data_indexes_.size()];
				batch.batch_data[itemid].original_data = provider_.get_data(idx);
			}

			// move to next data
			if (start_index_ + batch_size_ >= (int)data_indexes_.size()) {
				if (random_shuffle_) {
					std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
				}
				else {
					iter_++;
					srand(iter_);
					std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
				}

				// We have reached the end. Restart from the first.
				//LOG(INFO) << "Restarting data prefetching from start.";
			}

			start_index_ = (start_index_ + batch_size_) % (int)data_indexes_.size();
		}
	}
}