
#pragma once

#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>

namespace caffepro {
	class reduce_wrapper {
	public:
		// context could be null
		reduce_wrapper(caffepro_context *context = nullptr, int device_id = -1);

	public:
		data_type sum(const int n, const data_type *x);
		data_type sum2(const int n, const data_type *x);
		data_type max(const int n, const data_type *x);
		data_type min(const int n, const data_type *x);

	private:
		caffepro_context *context_;
		int device_id_;
	};
}