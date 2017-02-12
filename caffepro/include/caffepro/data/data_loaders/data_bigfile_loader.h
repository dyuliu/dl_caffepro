
#pragma once 

#include <caffepro/object_model/data_model/data_loader.h>
#include <string>
#include <map>

namespace caffepro {
	namespace data_model {
		class data_bigfile_loader : public data_loader {
		public:
			data_bigfile_loader(data_provider &provider);
			virtual ~data_bigfile_loader();

		public:
			virtual void load();

		protected:
			void load_bigfile(const std::string &bigfile_path, int label_id);
			void load_metadata(const std::string &metadata_file);

		protected:
			std::map<std::string, int> label2classID_;
			std::map<int, std::string> classID2label_;
		};
	}
}