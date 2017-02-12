
#pragma once

#include <fstream>
#include <string>
#include <cstdlib>

namespace caffepro {
	namespace data_utils {
		class binary_reader {
		private:
			std::istream &m_stream;

		public:
			binary_reader(std::istream &stream) : m_stream(stream) {}

			template <class T>
			T read() {
				T r;
				m_stream.read((char *)&r, sizeof(T));
				return r;
			}

			template <>
			std::string read<std::string>() {
				int len = read<int>();
				if (!m_stream) return std::string();

				char *buf = (char *)alloca(len + 1);
				m_stream.read(buf, len);
				buf[len] = '\0';

				return std::string(buf);
			}

			std::string read_fixedlen_string(int len) {
				char *buf = (char *)alloca(len);
				if (!m_stream.read(buf, len)) return std::string();

				return std::string(buf);
			}
		};

		class binary_writer {
		private:
			std::ostream &m_stream;

		public:
			binary_writer(std::ostream &stream) : m_stream(stream) {}

			template <class T>
			void write(const T v) {
				m_stream.write((const char *)&v, sizeof(T));
			}

			template <>
			void write<std::string>(const std::string v) {
				int len = (int)v.size();
				write(len);

				m_stream.write(v.c_str(), len);
			}

			void write_fixedlen_string(const std::string &v, int len) {
				char *buf = (char *)alloca(len);
				memset(buf, 0, len);
				memcpy(buf, v.c_str(), v.size());
				m_stream.write(buf, len);
			}
		};
	}
}