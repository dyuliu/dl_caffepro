
#pragma once

#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>

namespace caffepro {

	typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;
	
	template <class DTYPE>
	class cublas_wrapper {
	public:
		cublas_wrapper(caffepro_context *context, int device_id = -1) 
			: context_(context) {
			CHECK(context);

			if (device_id < 0) {
				device_id = context->get_current_device()->device_id();
			}

			device_id_ = device_id;
			handle_ = context->get_device(device_id)->cublas_handle();
		}

		void scal_addscalar(const int n, const DTYPE scal, const DTYPE shift, DTYPE *y);

		void fill_constant(const int n, const DTYPE value, DTYPE *y);

		static void fill_constant(int device_id, const int n, const DTYPE value, DTYPE *y);
		
		void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N, const DTYPE alpha, const DTYPE* A,
			const DTYPE* x, const DTYPE beta, DTYPE* y);
		
		void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K,
			const DTYPE alpha, const DTYPE* A, const DTYPE* B,
			const DTYPE beta, DTYPE* C);

		void axpy(const int n, const DTYPE alpha, const DTYPE* x, DTYPE* y);

		void scal(const int n, const DTYPE alpha, DTYPE *x);

		void scal_dev(const int n, const DTYPE *alpha, DTYPE *x);

		void axpby(const int n, const DTYPE alpha, const DTYPE* x, const DTYPE beta, DTYPE* y);

		void axpby_dev(const int n, const DTYPE *alpha, const DTYPE* x, const DTYPE beta, DTYPE* y);

		void axpby(const int n, const DTYPE alpha, const DTYPE* x, const int inc_x, const DTYPE beta, DTYPE* y, const int inc_y);

		void scale(const int n, const DTYPE alpha, const DTYPE *x, DTYPE *y);

		void scale_dev(const int n, const DTYPE *alpha, const DTYPE *x, DTYPE *y);

		void transpose(const int rows, const int cols, const DTYPE *x, DTYPE *y);

		void transpose_add(const int rows, const int cols, const DTYPE *x, const DTYPE beta, DTYPE *y);

		void dot(const int n, const DTYPE* x, const DTYPE* y, DTYPE* out);

		void mul(const int n, const DTYPE* a, const DTYPE* b, DTYPE* y);

		void max_scalar(const int n, const DTYPE min_v, DTYPE *y);

		void min_scalar(const int n, const DTYPE max_v, DTYPE *y);

		void add_scalar(const int n, const DTYPE alpha, DTYPE* y);

		void sqrt(const int n, const DTYPE *x, DTYPE *y);

		void copy(const int n, const DTYPE* x, DTYPE* y);

		void exp(const int n, const DTYPE* x, DTYPE* y);

		void batch2sp(const int n, const int len, const DTYPE *batch_data, const DTYPE *sp_pos, DTYPE *sp_data, const DTYPE scale_target);

		void sp2batch(const int n, const int len, const DTYPE *sp_data, const DTYPE *sp_pos, DTYPE *batch_data, const DTYPE scale_target);

		DTYPE asum(const int n, const DTYPE *x);

		DTYPE sum(const int n, const DTYPE *x);

		DTYPE sum2(const int n, const DTYPE *x);

	private:
		caffepro_context *context_;
		cublasHandle_t handle_;
		int device_id_;
	};
}