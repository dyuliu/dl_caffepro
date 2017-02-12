
#pragma once

#include <caffepro/object_model/data_model/data_entry.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>

namespace caffepro {
	namespace data_model {
		class rfcn_entry : public data_entry {
		public:
			// definations
			typedef std::pair<float, float> range_f;

		public:
			rfcn_entry(data_provider &provider, const std::string &name);
			~rfcn_entry();

		public:
			virtual void init();
			virtual void prepare(batch_descriptor &batch);

		protected:
			void select_proposal(const data_container &original_image, data_container &processed_image, int pos_neg_assign);
			
			void cluster_proposal(batch_descriptor &batch, std::vector<int> &target_widths, std::vector<int> &target_heights);

			void prepare_one(const data_container &original_image, data_container &processed_image, 
				int target_width, int target_height);

		protected:
			// attributes
			range_f pos_range_, neg_range_;
			int batch_size_;
			int batch_img_scale_, batch_img_min_scale_;
			int foreground_classes_;
			int max_scale_small_object_;
			int channel_num_;
			bool enable_flip_;
			float pos_ratio_, neg_ratio_;
			float padding_ratio_, padding_length_, src_padding_length_;
			float feature_map_padding_ratio_;
			float feature_map_start_, feature_map_scale_;
			float mean_value_;
		};
	}
}