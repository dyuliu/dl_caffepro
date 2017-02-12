
#pragma once

#include <caffepro/object_model/data_model/data_container.h>

namespace caffepro {
	namespace data_model {
		struct data_bigfile_container : public virtual data_container {
			size_t				data_length;					// data length in big file
			long long			data_offset;					// data offset in big file

			// prototype methods
			virtual data_container* create_new();
			virtual data_container* clone();

			// cache support
			virtual void load_data();
			virtual void unload_data();
		};
	}
}