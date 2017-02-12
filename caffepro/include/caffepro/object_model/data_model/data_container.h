
#pragma once 

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/utils/data_utils/box.h>
#include <caffepro/caffepro.h>
#include <boost/shared_ptr.hpp>
#include <boost/any.hpp>
#include <map>

namespace caffepro {
	namespace data_model {
		struct data_container : public caffepro_object {
			// definations 
			typedef cv::Mat								datum;
			typedef std::map<std::string, boost::any>	config_dic;

			// required attributes
			boost::shared_ptr<datum>		data;
			std::string						data_name;
			std::string						storage_path;
			std::string						class_name;
			int								label_id;

			// optional attributes
			// mainly for convenient purpose
			int								width;
			int								height;
			int								raw_width;
			int								raw_height;
			std::vector<int>				label_ids;
			std::vector<data_utils::Box>	groundtruth_boxes;
			std::vector<data_utils::Box>	proposal_boxes;
			config_dic						additional_data;

			data_container();
			virtual ~data_container();

			// prototype methods
			// DON'T FORGET to override in the inherited classes
			virtual data_container* create_new();
			virtual data_container* clone(); 

			// cache support
			// override if lazy-loading needed
			virtual void load_data() {}
			virtual void unload_data() {}

			DISABLE_COPY_AND_ASSIGN(data_container);
		};
	}
}