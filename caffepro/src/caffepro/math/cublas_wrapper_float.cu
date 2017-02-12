
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {

	template <typename DTYPE>
	__global__ void scal_addscalar_kernel(const int n, const DTYPE scal, const DTYPE shift, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = y[index] * scal + shift;
		}
	}

	template <>
	void cublas_wrapper<float>::scal_addscalar(const int n, const float scal, const float shift, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(scal_addscalar_kernel<float>, n)(n, scal, shift, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void fill_constant_kernel(const int n, const DTYPE value, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = value;
		}
	}

	template <>
	void cublas_wrapper<float>::fill_constant(const int n, const float value, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(fill_constant_kernel<float>, n)(n, value, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::fill_constant(int device_id, const int n, const float value, float *y) {
		ENTER_DEVICE_CONTEXT(device_id)
			KERNEL_CALL(fill_constant_kernel<float>, n)(n, value, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void mul_kernel(const int n, const DTYPE* a, const DTYPE* b, DTYPE* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = a[index] * b[index];
		}
	}

	template <>
	void cublas_wrapper<float>::mul(const int n, const float* a, const float* b, float* y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(mul_kernel<float>, n)(n, a, b, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void max_scalar_kernel(const int n, const DTYPE min_v, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = max(y[index], min_v);
		}
	}

	template <>
	void cublas_wrapper<float>::max_scalar(const int n, const float min_v, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(max_scalar_kernel, n)(n, min_v, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void min_scalar_kernel(const int n, const DTYPE max_v, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = min(y[index], max_v);
		}
	}

	template <>
	void cublas_wrapper<float>::min_scalar(const int n, const float max_v, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(min_scalar_kernel, n)(n, max_v, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void add_scalar_kernel(const int n, const DTYPE alpha, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] += alpha;
		}
	}

	template <>
	void cublas_wrapper<float>::add_scalar(const int n, const float alpha, float* y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(add_scalar_kernel, n)(n, alpha, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void scal_dev_kernel(const int n, const DTYPE *alpha, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			DTYPE v = *alpha;
			if (v == 0) {
				y[index] = 0;
			}
			else {
				y[index] *= v;
			}
		}
	}

	template <>
	void cublas_wrapper<float>::scal_dev(const int n, const float *alpha, float *x) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(scal_dev_kernel, n)(n, alpha, x);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void scale_dev_kernel(const int n, const DTYPE *alpha, const DTYPE *x, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			DTYPE v = *alpha;
			if (v == 0) {
				y[index] = 0;
			}
			else {
				y[index] = x[index] * v;
			}
		}
	}

	template <>
	void cublas_wrapper<float>::scale_dev(const int n, const float *alpha, const float *x, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(scale_dev_kernel, n)(n, alpha, x, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void axpby_dev_kernel(const int n, const DTYPE *alpha, const DTYPE *x, const DTYPE beta, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			if (beta == 0) {
				y[index] = x[index] * (*alpha);
			}
			else {
				y[index] = y[index] * beta + x[index] * (*alpha);
			}
		}
	}

	template <>
	void cublas_wrapper<float>::axpby_dev(const int n, const float *alpha, const float* x, const float beta, float* y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(axpby_dev_kernel, n)(n, alpha, x, beta, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void sqrt_kernel(const int n, const DTYPE *x, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = sqrt(max(x[index], (DTYPE)0.));
		}
	}

	template <>
	void cublas_wrapper<float>::sqrt(const int n, const float *x, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(sqrt_kernel<float>, n)(n, x, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void exp_kernel(const int n, const DTYPE *x, DTYPE *y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = exp(x[index]);
		}
	}

	template <>
	void cublas_wrapper<float>::exp(const int n, const float *x, float *y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(exp_kernel<float>, n)(n, x, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void batch2sp_kernel(const int n, const int len, const DTYPE *batch_data, const DTYPE *sp_pos, DTYPE *sp_data, const DTYPE scale_target) {
		CUDA_KERNEL_LOOP(index, n * len) {
			int cur_n = index / len, cur_k = index % len;
			int pos = (int)(sp_pos[cur_n] + 0.5);
			if (scale_target == 0) {
				sp_data[pos + cur_k] = batch_data[index];
			}
			else {
				sp_data[pos + cur_k] = sp_data[pos + cur_k] * scale_target + batch_data[index];
			}
		}
	}

	template <>
	void cublas_wrapper<float>::batch2sp(const int n, const int len, const float *batch_data, const float *sp_pos, float *sp_data, const float scale_target) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(batch2sp_kernel, n * len)(n, len, batch_data, sp_pos, sp_data, scale_target);
		EXIT_DEVICE_CONTEXT;
	}

	template <typename DTYPE>
	__global__ void sp2batch_kernel(const int n, const int len, const DTYPE *sp_data, const DTYPE *sp_pos, DTYPE *batch_data, const DTYPE scale_target) {
		CUDA_KERNEL_LOOP(index, n * len) {
			int cur_n = index / len, cur_k = index % len;
			int pos = (int)(sp_pos[cur_n] + 0.5);
			if (scale_target == 0) {
				batch_data[index] = sp_data[pos + cur_k];
			}
			else {
				batch_data[index] = batch_data[index] * scale_target + sp_data[pos + cur_k];
			}
		}
	}

	template <>
	void cublas_wrapper<float>::sp2batch(const int n, const int len, const float *sp_data, const float *sp_pos, float *batch_data, const float scale_target) {
		ENTER_DEVICE_CONTEXT(device_id_)
			KERNEL_CALL(sp2batch_kernel, n * len)(n, len, sp_data, sp_pos, batch_data, scale_target);
		EXIT_DEVICE_CONTEXT;
	}
}