
#pragma once

#include <caffepro/caffepro.h>

namespace caffepro {

	inline int calc_output_size(int input_size, int ksize, int stride, int pad, bool size_floor) {
		return size_floor
			? (input_size - ksize + 2 * pad) / stride + 1
			: (input_size - ksize + 2 * pad + stride - 1) / stride + 1;
	}

	inline bool should_bp(unsigned int selector, int target_index) {
		return selector & (1U << target_index) ? true : false;
	}

	inline bool should_release(unsigned int mask, int blob_index) {
		return mask & (1U << blob_index) ? true : false;
	}

	inline data_type get_beta(unsigned int selector, int target_index) {
		return (selector & (1U << target_index)) ? (data_type)0 : (data_type)1;
	}
}