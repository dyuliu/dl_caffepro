
#include <caffepro/data/data_accessors/sequential_proposal_accessor.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/caffepro_config.h>
#include <random>
#include <algorithm>

namespace caffepro {
	namespace data_model {
		sequential_proposal_accessor::sequential_proposal_accessor(data_provider &provider)
			: data_accessor(provider), batch_size_(0), start_index_(0), start_proposal_(0),
			start_view_(0), random_shuffle_(false), views_(1) {
			// nothing to do
		}

		sequential_proposal_accessor::~sequential_proposal_accessor() {
			// nothing to do
		}

		void sequential_proposal_accessor::init() {
			data_accessor::init();
			int data_size = (int)provider_.dataset_size();
			data_indexes_.resize(data_size);
			for (int i = 0; i < data_size; i++) {
				data_indexes_[i] = i;
			}
			caffepro_config_reader &config = provider_.data_provider_config();
			batch_size_ = config.get<int>("batch_size");
			random_shuffle_ = config.get<bool>("random_shuffle", false, false);
			if (config.get<bool>("include_flip", false, false)) {
				views_ = 2;
			}

			if (random_shuffle_) {
				srand(std::random_device()());
				std::random_shuffle(data_indexes_.begin(), data_indexes_.end());
			}
		}

		void sequential_proposal_accessor::get_batch(batch_descriptor &batch) {
			batch.batch_data.resize(batch_size_);

			for (int itemid = 0; itemid < batch_size_; itemid++) {
				int idx = data_indexes_[start_index_];
				batch.batch_data[itemid].original_data.reset(provider_.get_data(idx)->clone());
				auto &data = batch.batch_data[itemid].original_data;

				data->additional_data.erase("crop_box");
				data->additional_data.erase("flip");
				if (start_proposal_ < data->proposal_boxes.size()) {
					data->additional_data["crop_box"] = data->proposal_boxes[start_proposal_];
				}
				data->additional_data["flip"] = (start_view_ > 0);

				// move to next
				start_view_++;
				if (start_view_ >= views_) {
					start_view_ = 0;
					start_proposal_++;

					if (start_proposal_ >= data->proposal_boxes.size()) {
						start_proposal_ = 0;
						start_index_ = (start_index_ + 1) % data_indexes_.size();
					}
				}
			}
		}
	}
}