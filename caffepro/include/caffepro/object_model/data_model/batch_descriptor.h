
#pragma once 

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/object_model/data_model/data_container.h>

#include <boost/any.hpp>
#include <map>

namespace caffepro {
	namespace data_model {
		struct batch_descriptor : public caffepro_object {
			struct batch_datum {
				boost::shared_ptr<data_container>		original_data;
				boost::shared_ptr<data_container>		processed_data;
				data_container::config_dic				attachment;
			};

			// attribute
			std::vector<batch_datum>					batch_data;
			data_container::config_dic					batch_attribute;

			batch_descriptor();
			virtual ~batch_descriptor();

			// prototype methods
			// DON'T FORGET to override in the inherited classes
			virtual batch_descriptor* create_new();

			DISABLE_COPY_AND_ASSIGN(batch_descriptor);
		};
	}
}