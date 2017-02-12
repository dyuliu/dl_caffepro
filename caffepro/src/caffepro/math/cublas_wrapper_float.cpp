
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>

namespace caffepro {

	template <>
	void cublas_wrapper<float>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const float alpha, const float* A, const float* x,
		const float beta, float* y) {
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSgemv(handle_, cuTransA, N, M, &alpha,
				A, N, x, 1, &beta, y, 1));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
		const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, 
		const float beta, float* C) {
		// Note that cublas follows fortran order.
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSgemm(handle_, cuTransB, cuTransA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::axpy(const int n, const float alpha, const float* x, float* y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSaxpy(handle_, n, &alpha, x, 1, y, 1));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::scal(const int n, const float alpha, float *x) {
		ENTER_DEVICE_CONTEXT(device_id_)
			if (alpha == 0) {
				fill_constant(n, 0, x);
			}
			else {
				CUBLAS_CHECK(cublasSscal(handle_, n, &alpha, x, 1));
			}
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::axpby(const int n, const float alpha, const float* x,
		const float beta, float* y) {
		scal(n, beta, y);
		axpy(n, alpha, x, y);
	}

	template <>
	void cublas_wrapper<float>::axpby(const int n, const float alpha, const float* x, const int inc_x, const float beta, float* y, const int inc_y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSscal(handle_, n, &beta, y, inc_y));
			CUBLAS_CHECK(cublasSaxpy(handle_, n, &alpha, x, inc_x, y, inc_y));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::scale(const int n, const float alpha, const float *x, float* y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			if (x != y) {
				CUBLAS_CHECK(cublasScopy(handle_, n, x, 1, y, 1));
			}
			scal(n, alpha, y);
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::transpose(const int rows, const int cols, const float* x, float *y)
	{
		const float alpha = 1.f, beta = 0.f;
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols,
				&alpha, x, cols,
				&beta, NULL, rows,
				y, rows));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::transpose_add(const int rows, const int cols, const float* x, const float beta, float *y)
	{
		const float alpha = 1.f;
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols,
				&alpha, x, cols,
				&beta, y, rows,
				y, rows));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::dot(const int n, const float* x, const float* y, float* out) {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSdot(handle_, n, x, 1, y, 1, out));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	void cublas_wrapper<float>::copy(const int n, const float* x, float* y) {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUDA_CHECK(cudaMemcpyAsync(y, x, n * sizeof(float), cudaMemcpyDeviceToDevice));
		EXIT_DEVICE_CONTEXT;
	}

	template <>
	float cublas_wrapper<float>::asum(const int n, const float *x) {
		float v = 0.f;

		ENTER_DEVICE_CONTEXT(device_id_)
			CUBLAS_CHECK(cublasSasum(handle_, n, x, 1, &v));
		EXIT_DEVICE_CONTEXT;

		return v;
	}

	template <>
	float cublas_wrapper<float>::sum(const int n, const float *x) {
		float v = 0.f;

		ENTER_DEVICE_CONTEXT(device_id_)
			float *sum_mul = reinterpret_cast<float *>(
				context_->get_device(device_id_)->memory()->allocate(n * sizeof(float))
				);
			fill_constant(n, 1.f, sum_mul);
			dot(n, x, sum_mul, &v);
			context_->get_device(device_id_)->memory()->free(sum_mul);
		EXIT_DEVICE_CONTEXT;

		return v;
	}

	template <>
	float cublas_wrapper<float>::sum2(const int n, const float *x) {
		float v = 0.f;

		ENTER_DEVICE_CONTEXT(device_id_)
			float *x2 = reinterpret_cast<float *>(
				context_->get_device(device_id_)->memory()->allocate(n * sizeof(float))
				);
			mul(n, x, x, x2);
			v = asum(n, x2);
			context_->get_device(device_id_)->memory()->free(x2);
		EXIT_DEVICE_CONTEXT;

		return v;
	}
}