
#include <caffepro/data/data_accessors/sequential_rank_accessor.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <random>
#include <algorithm>

namespace caffepro {
	namespace data_model {
		sequential_rank_accessor::sequential_rank_accessor(data_provider &provider)
			: data_accessor(provider), batch_size_(0), start_index_(0), random_shuffle_(false) {
			// nothing to do
		}

		sequential_rank_accessor::~sequential_rank_accessor() {
			// nothing to do
		}

		void sequential_rank_accessor::init() {
			data_accessor::init();
			int data_size = (int)provider_.dataset_size();
			data_indexes_.resize(data_size);
			for (int i = 0; i < data_size; i++) {
				data_indexes_[i] = i;
			}
			caffepro_config_reader &config = provider_.data_provider_config();
			batch_size_ = config.get<int>("batch_size");
			ranking_batch_size_ = config.get<int>("ranking_batch_size");
			random_shuffle_ = config.get<bool>("random_shuffle", false, false);

			CHECK_LE(batch_size_, ranking_batch_size_);

			// init score map
			for (int i = 0; i < (int)provider_.dataset_size(); i++) {
				const std::string &data_name = provider_.get_data(i)->data_name;
				scores_[data_name] = FLT_MAX;
			}

			if (random_shuffle_) {
				srand(std::random_device()());
				std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
			}
		}

		void sequential_rank_accessor::get_batch(batch_descriptor &batch) {
			batch.batch_data.resize(batch_size_);

			std::vector<std::pair<data_type, int> > sel_data(ranking_batch_size_);
			for (int itemid = 0; itemid < ranking_batch_size_; itemid++) {
				int idx = data_indexes_[(start_index_ + itemid) % (int)data_indexes_.size()];
				const std::string &data_name = provider_.get_data(idx)->data_name;
				sel_data[itemid] = std::make_pair(scores_[data_name], itemid);
			}

			sort(sel_data.begin(), sel_data.end(), std::greater<std::pair<data_type, int> >());
			std::set<int> selected;

			// select batch_size_ from ranking_batch_size_ with larger scores
			for (int i = 0; i < batch_size_; i++) {
				int itemid = sel_data[i].second;
				selected.insert(itemid);
			}

			// fill batch data in original order
			for (int itemid = 0, i = 0; itemid < ranking_batch_size_; itemid++) {
				if (selected.count(itemid)) {
					int idx = data_indexes_[(start_index_ + itemid) % (int)data_indexes_.size()];
					batch.batch_data[i].original_data = provider_.get_data(idx);
					i++;
				}
			}

			// move to next data
			if (start_index_ + ranking_batch_size_ >= (int)data_indexes_.size()) {
				if (random_shuffle_) {
					std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
				}

				// We have reached the end. Restart from the first.
				//LOG(INFO) << "Restarting data prefetching from start.";
			}

			start_index_ = (start_index_ + ranking_batch_size_) % (int)data_indexes_.size();
		}

		void sequential_rank_accessor::update_score(const std::string &data_name, data_type new_score) {
			CHECK(scores_.count(data_name)) << "Invalid picture name: " << data_name;
			scores_[data_name] = new_score;
		}
	}
}