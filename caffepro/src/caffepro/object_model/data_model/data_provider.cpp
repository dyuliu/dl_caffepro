
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/context/common_names.h>

namespace caffepro {
	namespace data_model {
		using std::vector;
		using std::string;

		data_provider::data_provider(caffepro_context *context, caffepro_config *config) 
			: context_(context), config_(config), cache_data_(true), 
			img_info_(new analyzer_proto::Images), test_img_info_(new analyzer_proto::Images) {
			CHECK(context_);
			CHECK(config_) << "Config must be provided to initialize data provider";
			data_provider_config_.set_config(config_);
			data_prefetch_event_name_ = "data_provider_prefetch";
		}

		data_provider::~data_provider() {
			finish_prefetch();
		}

		void data_provider::init() {
			cache_data_ = data_provider_config_.get<bool>("cache_data", false, true);

			// loading data
			for (auto iter = data_loaders_.begin(); iter != data_loaders_.end(); ++iter) {
				(*iter)->load();
			}

			// init data accessor
			if (data_accessor_) {
				data_accessor_->init();
			}

			// init data entries
			for (auto iter = data_entries_.begin(); iter != data_entries_.end(); ++iter) {
				(*iter)->init();
			}

			// prefetch for the first batch
			begin_prefetch();
		}

		void data_provider::begin_prefetch() {
			finish_prefetch();

			context_->events()->create(data_prefetch_event_name_,
				event_manager::EVENT_TYPE_PREPARE_BATCH, data_prefetch, this);
		}

		void* data_provider::data_prefetch(void* layer_pointer) {
			data_provider *ins = static_cast<data_provider *>(layer_pointer);

			ins->prefetch_batch_.reset(ins->create_batch_descriptor());
			
			if (ins->data_accessor_) {
				ins->data_accessor_->get_batch(*(ins->prefetch_batch_.get()));
			}

			if (!ins->cache_data_) {
				for (auto iter = ins->prefetch_batch_->batch_data.begin(); iter != ins->prefetch_batch_->batch_data.end(); ++iter) {
					iter->original_data->load_data();
				}
			}

			// make a copy for processed data
			for (auto iter = ins->prefetch_batch_->batch_data.begin(); iter != ins->prefetch_batch_->batch_data.end(); ++iter) {
				iter->processed_data.reset(iter->original_data->clone());
			}

			for (auto iter = ins->data_entries_.begin(); iter != ins->data_entries_.end(); ++iter) {
				(*iter)->prepare(*(ins->prefetch_batch_.get()));
			}

			if (!ins->cache_data_) {
				for (auto iter = ins->prefetch_batch_->batch_data.begin(); iter != ins->prefetch_batch_->batch_data.end(); ++iter) {
					iter->original_data->unload_data();
				}
			}


			return nullptr;
		}

		void data_provider::finish_prefetch() {
			context_->events()->wait(data_prefetch_event_name_);
		}

		void data_provider::forward() {
			finish_prefetch();

			
			current_batch_ = prefetch_batch_;
			
			// To do: add img indexes into .info - which need global info instance
			for (auto iter = current_batch_->batch_data.begin(); iter != current_batch_->batch_data.end(); ++iter) {
				if (context_->get_phase() == caffepro_context::TRAIN) {
					auto img = img_info_->add_images();
					img->set_class_name(iter->processed_data->class_name);
					img->set_file_name(iter->processed_data->data_name);
					img->set_label_id(iter->processed_data->label_id);
				}
				else if (context_->get_phase() == caffepro_context::TEST){
					auto img = test_img_info_->add_images();
					img->set_class_name(iter->processed_data->class_name);
					img->set_file_name(iter->processed_data->data_name);
					img->set_label_id(iter->processed_data->label_id);
				}
			}

			for (auto iter = data_entries_.begin(); iter != data_entries_.end(); ++iter) {
				(*iter)->forward();
			}

			caffepro_object *last_datasource = context_->set_shared_object(namespace_, SHAREDOBJNAME_DATASOURCE, this);
			if (last_datasource != nullptr) {
				if (!dynamic_cast<data_provider *>(last_datasource)) {
					LOG(FATAL) << "Only one data source is allowed in the namespace: " << namespace_;
				}
				else if (last_datasource != this) {
					LOG(ERROR) << "Warning: more than one data sources in the same namespace: " << namespace_;
				}
			}

		}

		void data_provider::add_data(boost::shared_ptr<data_container> data) {
			CHECK(!database_.dataset_index.count(data->data_name))
				<< "Error: more than one data item shares the same name: " << data->data_name;

			database_.dataset.push_back(data);
			database_.dataset_index[data->data_name] = data;
		}

		void data_provider::add_entry(boost::shared_ptr<data_entry> entry) {
			std::set<string> types;
			for (auto iter = data_entries_.begin(); iter != data_entries_.end(); ++iter) {
				types.insert((*iter)->type());
			}

			for (auto iter = entry->required_entries().begin(); iter != entry->required_entries().end(); ++iter) {
				CHECK(types.count(*iter)) << "Error: missing required entry: " << *iter;
			}

			data_entries_.push_back(entry);
		}
	}
}