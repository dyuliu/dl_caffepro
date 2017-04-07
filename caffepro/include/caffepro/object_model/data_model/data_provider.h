
#pragma once 

#include <caffepro/proto/analyzer_proto.pb.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>
#include <caffepro/object_model/data_model/data_loader.h>
#include <caffepro/object_model/data_model/data_accessor.h>
#include <caffepro/object_model/data_model/data_entry.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/context/caffepro_context.h>

namespace caffepro {
	namespace data_model {

		class data_provider : public caffepro_object {
		public:
			// definations 
			struct database {
				std::vector<boost::shared_ptr<data_container> > dataset;
				std::map<std::string, boost::shared_ptr<data_container> > dataset_index;
				data_container::config_dic metadata;
			};

		public:
			// fetch functions

			// when dumped, should be cleared
			boost::shared_ptr<analyzer_proto::Images> &img_info() { return img_info_; }
			boost::shared_ptr<analyzer_proto::Images> &test_img_info() { return test_img_info_; }


			// prototype related
			void set_datacontainer_proto(boost::shared_ptr<data_container> prototype) {
				CHECK(prototype);
				data_container_prototype_ = prototype;
			}
			void set_batch_descriptor_proto(boost::shared_ptr<batch_descriptor> prototype) {
				CHECK(prototype);
				batch_descriptor_prototype_ = prototype;
			}
			data_container *create_data_container() { return data_container_prototype_->create_new(); }
			batch_descriptor *create_batch_descriptor() { return batch_descriptor_prototype_->create_new(); }

			// database related
			database &get_database() { return database_; }
			const database &get_database() const { return database_; }
			size_t dataset_size() const { return database_.dataset.size(); }
			boost::shared_ptr<data_container> get_data(int index) const { return database_.dataset[index]; }
			boost::shared_ptr<data_container> get_data(const std::string &data_name) const { 
				return database_.dataset_index.count(data_name) ? database_.dataset_index.find(data_name)->second : nullptr; 
			}
			void clear_data() { database_.dataset.clear(); database_.dataset_index.clear(); }
			boost::any get_metadata(const std::string &name) const { 
				return database_.metadata.count(name) ? database_.metadata.find(name)->second : boost::any(); 
			}
			boost::any set_metadata(const std::string &name, boost::any value) {
				boost::any last = database_.metadata[name];
				database_.metadata[name] = value;
				return last;
			}
			void clear_metadata() { database_.metadata.clear(); }

			// common
			caffepro_context *context() const { return context_; }
			caffepro_config *config() const { return config_; }
			caffepro_config_reader &data_provider_config() { return data_provider_config_; }
			const caffepro_config_reader &data_provider_config() const { return data_provider_config_; }
			void set_data_provider_name(const std::string &name) { data_provider_config_.set_default_section_name(name); }
			boost::shared_ptr<batch_descriptor> prefetch_batch() const { return prefetch_batch_; }
			boost::shared_ptr<batch_descriptor> current_batch() const { return current_batch_; }

			// components
			// data loader
			std::vector<boost::shared_ptr<data_loader> > &data_loaders() { return data_loaders_; }
			const std::vector<boost::shared_ptr<data_loader> > &data_loaders() const { return data_loaders_; }
			size_t data_loader_size() const { return data_loaders_.size(); }
			void add_data_loader(boost::shared_ptr<data_loader> loader) { return data_loaders_.push_back(loader); }
			void clear_data_loader() { data_loaders_.clear(); }
			const std::string &get_namespace() const { return namespace_; }
			void set_namespace(const std::string &ns) { namespace_ = ns; }

			// data accessor
			boost::shared_ptr<data_accessor> get_data_accessor() const { return data_accessor_; }
			void set_data_accessor(boost::shared_ptr<data_accessor> accessor) { data_accessor_ = accessor; }

			// entry
			std::vector<boost::shared_ptr<data_entry> > &data_entries() { return data_entries_; }
			const std::vector<boost::shared_ptr<data_entry> > &data_entries() const { return data_entries_; }
			const boost::shared_ptr<data_entry> get_entry(const std::string &entry_name) const {
				for (auto iter = data_entries_.begin(); iter != data_entries_.end(); ++iter) {
					if ((*iter)->name() == entry_name) return *iter;
				}
				return nullptr;
			}
			boost::shared_ptr<data_entry> get_entry(const std::string &entry_name) {
				for (auto iter = data_entries_.begin(); iter != data_entries_.end(); ++iter) {
					if ((*iter)->name() == entry_name) return *iter;
				}
				return nullptr;
			}

		public:
			data_provider(caffepro_context *context, caffepro_config *config);
			virtual ~data_provider();

		public:
			// interfaces
			virtual void init();
			virtual void begin_prefetch();
			virtual void finish_prefetch();
			virtual void forward();

			// database related
			void add_data(boost::shared_ptr<data_container> data);

			// components related
			void add_entry(boost::shared_ptr<data_entry> entry);

		public:
			// factory functions
			virtual void auto_build();
			virtual data_entry *create_entry(const std::string &type, const std::string &name);

		protected:
			// prefetch
			static void* data_prefetch(void* layer_pointer);

		protected:
			// common
			caffepro_context *context_;
			caffepro_config *config_;
			caffepro_config_reader data_provider_config_;
			std::string data_prefetch_event_name_;
			std::string namespace_;

			// data prototype
			boost::shared_ptr<data_container> data_container_prototype_;
			boost::shared_ptr<batch_descriptor> batch_descriptor_prototype_;
			
			// database
			database database_;

			// batch
			boost::shared_ptr<batch_descriptor> prefetch_batch_, current_batch_;

			// components
			std::vector<boost::shared_ptr<data_loader> > data_loaders_;
			boost::shared_ptr<data_accessor> data_accessor_;
			std::vector<boost::shared_ptr<data_entry> > data_entries_;

			// other
			bool cache_data_; // whether caching data in memory

			// write img info of a batch into info
			boost::shared_ptr<analyzer_proto::Images> img_info_;
			boost::shared_ptr<analyzer_proto::Images> test_img_info_;


		private:
			DISABLE_COPY_AND_ASSIGN(data_provider);
		};
	}
}