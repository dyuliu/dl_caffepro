
#pragma once

#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>

namespace caffepro {

	void find_max_min(const int count, const data_type* data, data_type& output_max, data_type& output_min);

	void find_max_min(const int count, const unsigned int* data, unsigned int& output_max, unsigned int& output_min);
}