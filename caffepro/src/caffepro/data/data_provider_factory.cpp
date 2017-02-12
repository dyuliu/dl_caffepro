
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/data/data_containers/data_containers.h>
#include <caffepro/data/data_loaders/data_loaders.h>
#include <caffepro/data/data_accessors/data_accessors.h>
#include <caffepro/data/data_entries/data_entries.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/object_model/data_model/batch_descriptor.h>
#include <caffepro/object_model/caffepro_config.h>

namespace caffepro {
	namespace data_model {
		using std::string;
		using std::vector;

		data_entry *data_provider::create_entry(const std::string &type, const std::string &name) {
			data_entry *entry = nullptr;
			if (type == "image") {
				entry = new image_entry(*this, name);
			}
			else if (type == "label") {
				entry = new label_entry(*this, name);
			}
			else if (type == "rcnn") {
				entry = new rcnn_entry(*this, name);
			}
			else if (type == "rfcn") {
				entry = new rfcn_entry(*this, name);
			}
			else if (type == "image_sr") {
				entry = new image_sr_entry(*this, name);
			}
			else {
				LOG(FATAL) << "Unknown data entry name: " << type;
			}

			CHECK_EQ(type, entry->type());
			return entry;
		}

		void data_provider::auto_build() {
			// set data container
			set_datacontainer_proto(boost::shared_ptr<data_container>(new data_bigfile_container()));
			set_batch_descriptor_proto(boost::shared_ptr<batch_descriptor>(new batch_descriptor()));

			// add data loaders
			add_data_loader(boost::shared_ptr<data_loader>(new data_bigfile_loader(*this)));
				
			if (data_provider_config_.exist("color_kl_dir")) {
				add_data_loader(boost::shared_ptr<data_loader>(new kl_data_loader(*this)));
			}

			if (data_provider_config_.exist("multilabel_def_file")) {
				add_data_loader(boost::shared_ptr<data_loader>(new multilabel_loader(*this)));
			}

			if (data_provider_config_.exist("crop_box_file")) {
				add_data_loader(boost::shared_ptr<data_loader>(new box_proposal_loader(*this)));
			}

			// set data accessor
			set_data_accessor(boost::shared_ptr<data_accessor>(new sequential_accessor(*this)));

			// add data entries
			vector<string> entry_names = data_provider_config_.get_array<string>("entries", false);

			for (auto iter = entry_names.begin(); iter != entry_names.end(); ++iter) {
				CHECK(config_->section_exist(*iter));
				const string type = config_->get(*iter).get("type");
				add_entry(boost::shared_ptr<data_entry>(create_entry(type, *iter)));
			}
		}
	}
}