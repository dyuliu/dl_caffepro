
#pragma once

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/utils/string_uitls.h>
#include <caffepro/caffepro.h>
#include <map>
#include <fstream>

namespace caffepro {
	class caffepro_config : public caffepro_object {
	public:
		class config_section {
		public:
			config_section();
			~config_section();

		public:
			// interfaces
			bool add(const std::string &key, const std::string &value);

			// YICHEN: is required and default_value necessary?
			const std::string& get(
				const std::string &key,
				bool required = true,
				const std::string &default_value = ""
				) const;

		public:
			// fetch functions
			bool exist(const std::string &key) const { return data_.count(key) > 0; }

		private:
			std::map<std::string, std::string> data_;
		};

	public:
		caffepro_config();
		~caffepro_config();

	public:
		// interfaces
		void load_config(const std::string &filename);
		void load_config(std::istream &in);
		const config_section &get(const std::string &sec_name) const;
		const config_section &get() const; // get global config

	public:
		// fetch functions
		const config_section &operator[] (const std::string &sec_name) const { return get(sec_name); }
		const config_section &operator() () const { return get(); }
		bool section_exist(const std::string &sec_name) const { return sections_.count(sec_name) > 0; }

	protected:
		std::map<std::string, config_section> sections_;
		config_section empty_section_;

	private:
		DISABLE_COPY_AND_ASSIGN(caffepro_config);
	};

	class caffepro_config_reader : public caffepro_object {
	public:
		caffepro_config_reader();
		caffepro_config_reader(const caffepro_config &config, const std::string &default_sec_name);

	public:
		// interfaces
		// YICHEN: we may refactor the code into
		// 1. a function that maps key -> string value (it may fail and can return success status)
		// 2. functions that convert string value to a single or array of target value. They should be in strings_utils.h
		// is parameter required and default value necessary? default_value can always be assigned outside
		template <typename TargetType>
		TargetType get(const std::string &key, bool required = true, const TargetType &default_value = TargetType()) const {
			CHECK(valid());

			std::string sec_name = "";
			if (config_->section_exist(default_sec_name_)
				&& config_->get(default_sec_name_).exist(key)) {
				sec_name = default_sec_name_;
			}

			auto &sec = config_->get(sec_name);
			if (!sec.exist(key)) {
				if (required) {
					LOG(FATAL) << "Missing required field: " << key;
				}
				else {
					return default_value;
				}
			}

			const std::string &value = sec.get(key);
			TargetType result;
			std::stringstream sstream;
			sstream << value;
			sstream >> result;
			if (sstream.fail()) {
				LOG(FATAL) << "Bad type for the config item: (" << key << " - " << value << ")";
			}
			return result;
		}

		template <typename TargetType>
		std::vector<TargetType> get_array(const std::string &key, bool required = true) const {
			CHECK(valid());
			std::vector<TargetType> result;

			std::string sec_name = "";
			if (config_->section_exist(default_sec_name_)
				&& config_->get(default_sec_name_).exist(key)) {
				sec_name = default_sec_name_;
			}

			auto &sec = config_->get(sec_name);
			if (!sec.exist(key)) {
				if (required) {
					LOG(FATAL) << "Missing required field: " << key;
				}
				else {
					return result;
				}
			}

			const std::string &value = sec.get(key);
			std::vector<std::string> values = split(value, ',');
			
			for (int i = 0; i < (int)values.size(); i++) {
				std::stringstream sstream;
				sstream << trim(values[i]);
				TargetType r;
				sstream >> r;
				result.push_back(r);
			}
			return result;
		}

		bool exist(const std::string &key) {
			CHECK(valid());
			std::string sec_name = "";
			if (config_->section_exist(default_sec_name_)
				&& config_->get(default_sec_name_).exist(key)) {
				sec_name = default_sec_name_;
			}

			auto &sec = config_->get(sec_name);
			return sec.exist(key);
		}

	public:
		// fetch functions
		const caffepro_config *get_config() const { return config_; }
		void set_config(const caffepro_config *config) { config_ = config; }
		void set_default_section_name(const std::string &name) { default_sec_name_ = name; }
		const std::string &get_default_section_name(const std::string &name) const { return default_sec_name_; }
		bool valid() const { return config_ != nullptr; }

	private:
		const caffepro_config *config_;
		std::string default_sec_name_;
	};
}