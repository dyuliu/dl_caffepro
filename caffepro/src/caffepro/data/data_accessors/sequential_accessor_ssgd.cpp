
#include <caffepro/data/data_accessors/sequential_accessor_ssgd.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <random>
#include <algorithm>

namespace caffepro {
	namespace data_model {
		sequential_accessor_ssgd::sequential_accessor_ssgd(data_provider &provider)
			: data_accessor(provider), batch_size_(0), start_index_(0), random_shuffle_(false), iter_(0) {
			// nothing to do
			multinode_ = multinode::get();
			worker_id_ = multinode_->get_worker_id();
			worker_size_ = multinode_->get_worker_size();
		}

		sequential_accessor_ssgd::~sequential_accessor_ssgd() {
			// nothing to do
			delete[] worker_sequence_;
		}

		void sequential_accessor_ssgd::init() {
			data_accessor::init();

			// prepare
			int data_size = (int)provider_.dataset_size();
			data_indexes_.resize(data_size);
			caffepro_config_reader &config = provider_.data_provider_config();
			batch_size_ = config.get<int>("batch_size");
			random_shuffle_ = config.get<bool>("random_shuffle", false, false);

			// setting start index and end indx for all workers
			worker_sequence_ = new data_type[data_size];
			worker_data_size_ = (int)(data_size / worker_size_);
			worker_start_idx_ = worker_data_size_*worker_id_;
			worker_end_idx_ = worker_data_size_*(worker_id_ + 1);
			
			// just ID-0 machine do shuffle operation
			multinode_shffule();
		}

		void sequential_accessor_ssgd::multinode_shffule() {
			int data_size = (int)provider_.dataset_size();
			memset(worker_sequence_, 0, sizeof(data_type)*data_size);
			if (worker_id_ == 0) {
				for (int i = 0; i < data_size; i++) {
					data_indexes_[i] = i;
				}
				if (random_shuffle_) {
					srand(std::random_device()());
					std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
				}
				else {
					iter_++;
					srand(iter_);
					std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
				}
				for (int i = 0; i < data_size; i++) {
					worker_sequence_[i] = (data_type)data_indexes_[i];
				}
			}

			// synchronize
			multinode_->all_sum(worker_sequence_, data_size);
			for (int i = 0; i < data_size; i++) {
				data_indexes_[i] = (int)worker_sequence_[i];
			}
		}

		void sum(std::vector<int> &index_, int start_, int batch_size_) {
			int sum = 0;
			for (int i = start_; i < start_ + batch_size_; i++)
				sum += index_[i];
			COUT_CHEK << "Batch start in " << start_ << ", sum is " << sum << std::endl;
		}

		void sequential_accessor_ssgd::get_batch(batch_descriptor &batch) {
			batch.batch_data.resize(batch_size_);
			int idx = 0;
			for (int itemid = 0; itemid < batch_size_; itemid++) {
				if (worker_start_idx_ + itemid < worker_end_idx_) {
					idx = data_indexes_[worker_start_idx_ + itemid];
				}
				else {
					idx = data_indexes_[worker_start_idx_ - worker_data_size_ + itemid];
				}
				batch.batch_data[itemid].original_data = provider_.get_data(idx);
			}

			// reach end
			if (worker_start_idx_ + batch_size_ >= worker_end_idx_) {
				// if someone worker finish a epoch, waiting the rank-0 synchronize re-shuffle
				multinode_shffule();
				// reset cursor
				worker_start_idx_ = worker_data_size_*worker_id_;
			}
			// not reach end -> moving cursor
			else if (worker_start_idx_ + batch_size_ < worker_end_idx_) {
				worker_start_idx_ = worker_start_idx_ + batch_size_;
			}
		}
	}
}