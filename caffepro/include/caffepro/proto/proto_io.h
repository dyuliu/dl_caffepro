
#pragma once

#include <google/protobuf/message.h>

namespace caffepro {
	class proto_io {
	public:
		proto_io(google::protobuf::Message &proto);

	public:
		void from_text_file(const std::string &filename);
		void to_text_file(const std::string &filename);
		void from_binary_file(const std::string &filename);
		void to_binary_file(const std::string &filename);

	private:
		google::protobuf::Message &msg_;
	};
}