
#pragma once 

#include <caffepro/object_model/data_model/data_entry.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>

namespace caffepro {
	namespace data_model {
		class rcnn_entry : public data_entry {
		public:
			// definations
			typedef std::pair<float, float> range_f;

		public:
			rcnn_entry(data_provider &provider, const std::string &name);
			~rcnn_entry();

		public:
			virtual void init();
			virtual void prepare(batch_descriptor &batch);
		
		protected:
			void prepare_one(const data_container &original_image, data_container &processed_image, int pos_neg_assign);

		protected:
			// attributes
			range_f pos_range_, neg_range_;
			int batch_size_;
			int batch_img_size_;
			int foreground_classes_;
			int max_length_small_object_;
			int channel_num_;
			bool enable_flip_;
			float pos_ratio_, neg_ratio_;
			float padding_ratio_;
			int padding_length_;
			float mean_value_;
			std::string method_;
		};
	}
}