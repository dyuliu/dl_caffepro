
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4127 4244 4267 4996)
#endif

#include <caffepro/proto/proto_io.h>
#include <caffepro/caffepro.h>

#include <stdint.h>
#include <fcntl.h>
#include <io.h>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>

using std::string;
using std::ios;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::IstreamInputStream;

namespace caffepro {
	proto_io::proto_io(google::protobuf::Message &proto)
		: msg_(proto) {
	}

	void proto_io::from_text_file(const std::string &filename) {
		int fd = open(filename.c_str(), O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		FileInputStream* input = new FileInputStream(fd);
		CHECK(google::protobuf::TextFormat::Parse(input, &msg_));
		delete input;
		close(fd);
	}

	void proto_io::to_text_file(const std::string &filename) {
		int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
		FileOutputStream* output = new FileOutputStream(fd);
		CHECK(google::protobuf::TextFormat::Print(msg_, output));
		delete output;
		close(fd);
	}

	void proto_io::from_binary_file(const std::string &filename) {
		std::ifstream in(filename, std::ios::in | std::ios::binary);
		CHECK(in) << "File not found: " << filename;

		ZeroCopyInputStream* raw_input = new IstreamInputStream(&in);
		CodedInputStream* coded_input = new CodedInputStream(raw_input);
		coded_input->SetTotalBytesLimit(1024 * 1024 * 1024 + 1024 * 1024 * 512, 512 * 1024 * 1024);

		CHECK(msg_.ParseFromCodedStream(coded_input));

		delete coded_input;
		delete raw_input;
		in.close();
	}

	void proto_io::to_binary_file(const std::string &filename) {
		std::fstream output(filename, ios::out | ios::trunc | ios::binary);
		CHECK(msg_.SerializeToOstream(&output));
	}
}

#ifdef _MSC_VER
#  pragma warning(pop)
#endif