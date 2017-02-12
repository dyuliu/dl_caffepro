
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/utils/string_uitls.h>

namespace caffepro {
	using std::string;

	caffepro_config::caffepro_config() {
		// nothing to do
	}

	caffepro_config::~caffepro_config() {
		// nothing to do
	}

	void caffepro_config::load_config(const std::string &filename) {
		std::ifstream in(filename);
		if (!in) {
			LOG(FATAL) << "Unable to open config file: " << filename;
		}
		load_config(in);
	}

	void caffepro_config::load_config(std::istream &in) {
		string line_buffer;
		string current_section_name;

		while (std::getline(in, line_buffer)) {
			line_buffer = trim(line_buffer);

			if (!line_buffer.empty() && line_buffer[0] != '#') {
				if (line_buffer.front() == '[' && line_buffer.back() == ']') {
					current_section_name = trim(line_buffer.substr(1, (int)line_buffer.size() - 2));
				}
				else {
					int pos = (int)line_buffer.find_first_of('=');
					CHECK_NE(pos, string::npos) << "Bad format of config file: " << line_buffer;
					string key = trim(line_buffer.substr(0, pos));
					CHECK(!key.empty()) << "Bad format of config file: " << line_buffer;
					string value = trim(line_buffer.substr(pos + 1, (int)line_buffer.size() - pos - 1));
					if (!sections_[current_section_name].add(key, value)) {
						LOG(ERROR) << "Warning: duplicate key name in the config file: " << line_buffer;
					}
				}
			}
		}
	}

	const caffepro_config::config_section &caffepro_config::get(const std::string &sec_name) const {
		if (sections_.count(sec_name))  {
			return sections_.find(sec_name)->second;
		}
		else {
			return empty_section_;
		}
	}

	const caffepro_config::config_section &caffepro_config::get() const {
		return get("");
	}

	caffepro_config::config_section::config_section() {
		// nothing to do
	}

	caffepro_config::config_section::~config_section() {
		// nothing to do
	}


	bool caffepro_config::config_section::add(const std::string &key, const std::string &value) {
		bool r = (data_.count(key) == 0);
		data_[key] = value;

		return r;
	}

	const string& caffepro_config::config_section::get(const string &key, bool required,
		const string &default_value) const {
		if (!data_.count(key)) {
			if (required) {
				LOG(FATAL) << "Missing required field: " << key;
			}
			else {
				return default_value;
			}
		}

		return data_.find(key)->second;
	}

	caffepro_config_reader::caffepro_config_reader() 
		: config_(nullptr) {
	}
	
	caffepro_config_reader::caffepro_config_reader(const caffepro_config &config, const std::string &default_sec_name)
		: config_(&config), default_sec_name_(default_sec_name) {
	}
}