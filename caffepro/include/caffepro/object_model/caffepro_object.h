
#pragma once

#include <string>

namespace caffepro {

	class caffepro_object {

	public:
		// definition
		struct output_metric {
			std::string source;
			std::string name;
			double value;
		};

	protected:
		caffepro_object() {}

	public:
		virtual ~caffepro_object() {}
	};
}