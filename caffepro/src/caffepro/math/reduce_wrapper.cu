
#include <caffepro/math/reduce_wrapper.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace caffepro {
	template <typename T>
	struct square {
		__host__ __device__
		T operator()(const T& x) const {
			return x * x;
		}
	};

	reduce_wrapper::reduce_wrapper(caffepro_context *context, int device_id)
		: context_(context) {

		if (device_id < 0) {
			CUDA_CHECK(cudaGetDevice(&device_id));
		}

		device_id_ = device_id;
	}

	data_type reduce_wrapper::sum(const int n, const data_type *x) {
		data_type v = 0.f;
		ENTER_DEVICE_CONTEXT(device_id_)
			v = thrust::reduce(thrust::device, x, x + n, (data_type)0.f, thrust::plus<data_type>());
		EXIT_DEVICE_CONTEXT;
		return v;
	}

	data_type reduce_wrapper::sum2(const int n, const data_type *x) {
		data_type v = 0.f;
		ENTER_DEVICE_CONTEXT(device_id_)
			v = thrust::transform_reduce(thrust::device, x, x + n, square<data_type>(), (data_type)0.f, thrust::plus<data_type>());
		EXIT_DEVICE_CONTEXT;
		return v;
	}

	data_type reduce_wrapper::max(const int n, const data_type *x) {
		data_type v = 0.f;
		ENTER_DEVICE_CONTEXT(device_id_)
			v = thrust::reduce(thrust::device, x, x + n, -FLT_MAX, thrust::maximum<data_type>());
		EXIT_DEVICE_CONTEXT;
		return v;
	}

	data_type reduce_wrapper::min(const int n, const data_type *x) {
		data_type v = 0.f;
		ENTER_DEVICE_CONTEXT(device_id_)
			v = thrust::reduce(thrust::device, x, x + n, FLT_MAX, thrust::minimum<data_type>());
		EXIT_DEVICE_CONTEXT;
		return v;
	}
}