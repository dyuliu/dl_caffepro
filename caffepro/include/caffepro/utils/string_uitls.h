
#pragma once

#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace caffepro {
	inline bool is_space(char c) {
		return c == ' ' || c == '\r' || c == '\t' || c == '\n' || c == '\0';
	}

	inline std::string to_upper(std::string s) {
		for (size_t i = 0; i < s.size(); i++) {
			s[i] = (char)toupper(s[i]);
		}
		return s;
	}

	inline std::string trim(std::string s) {

		for (int i = (int)s.size() - 1; i >= 0 && is_space(s[i]); i--) {
			s.pop_back();
		}

		int start_pos = 0;
		for (; start_pos < s.size() && is_space(s[start_pos]); start_pos++);

		return s.substr(start_pos);
	}

	inline std::vector<std::string> split(std::string s, char sp = 0) {
		std::set<char> split;

		if (sp == 0) {
			split.insert(' '); split.insert('\r'); split.insert('\t'); split.insert('\n'); split.insert('\0');
		}
		else {
			split.insert(sp);
		}

		std::vector<std::string> result;
		std::string buffer;

		for (int i = 0; i < (int)s.size(); i++) {
			if (split.count(s[i])) {
				if (!buffer.empty()) {
					result.push_back(buffer);
					buffer.clear();
				}
			}
			else {
				buffer.push_back(s[i]);
			}
		}

		if (!buffer.empty()) result.push_back(buffer);

		return result;
	}

	inline bool string_equal_ignorecase(std::string s1, std::string s2) {
		s1 = to_upper(s1);
		s2 = to_upper(s2);
		return s1 == s2;
	}

	inline bool endwith_ignorecase(std::string s, std::string tail) {
		if (s.size() >= tail.size()) {
			return string_equal_ignorecase(s.substr(s.size() - tail.size()), tail);
		}
		else {
			return false;
		}
	}

	inline int string_to_int(std::string s) {
		std::stringstream sstream(s);
		int r;
		sstream >> r;
		return r;
	}

	inline float string_to_float(std::string s) {
		std::stringstream sstream(s);
		float r;
		sstream >> r;
		return r;
	}

	inline std::string int_to_string(int n) {
		std::stringstream s;
		std::string str;
		s << n;
		s >> str;
		return str;
	}

	inline std::string fill_zero(int val, int pad) {
		std::stringstream ss;
		ss << std::setw(pad) << std::setfill('0') << val;
		return ss.str();
	}
}