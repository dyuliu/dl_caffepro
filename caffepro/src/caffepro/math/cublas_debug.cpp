
#pragma once

#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>
#include <caffepro/math/cublas_debug.h>

namespace caffepro {

	void find_max_min(const int count, const data_type* data, data_type& output_max, data_type& output_min){
		output_max = -FLT_MAX;
		output_min = FLT_MAX;
		for (int i = 0; i < count; i++){
			output_max = fmax(output_max, data[i]);
			output_min = fmin(output_min, data[i]);
		}
	}

	void find_max_min(const int count, const unsigned int* data, unsigned int& output_max, unsigned int& output_min){
		output_max = 0;
		output_min = UINT_MAX;
		for (int i = 0; i < count; i++){
			output_max = (output_max > data[i]) ? output_max : data[i];
			output_min = (output_min < data[i]) ? output_min : data[i];
		}
	}
}