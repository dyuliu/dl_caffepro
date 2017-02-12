
#pragma once

#include <caffepro/proto/caffe.pb.h>
#include <caffepro/object_model/device_blob.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	class filler {
	public:
		explicit filler(caffepro_context *context, const FillerParameter& param) : 
			context_(context), filler_param_(param) {
			CHECK(context); 
		}

		virtual ~filler() {}
		virtual void fill(device_blob &blob) = 0;
		static filler* create(caffepro_context *context, const FillerParameter& param);

	protected:
		FillerParameter filler_param_;
		caffepro_context *context_;
	};

	class constant_filler : public filler {
	public:
		explicit constant_filler(caffepro_context *context, const FillerParameter& param)
			: filler(context, param) {}

		virtual void fill(device_blob &blob) {
			const data_type value = this->filler_param_.value();
			
			if (value == 0) {
				blob.dev_data_fill_zeros();
			}
			else {
				blob.fill_data(value);
			}
		}
	};

	class uniform_filler : public filler {
	public:
		explicit uniform_filler(caffepro_context *context, const FillerParameter& param)
			: filler(context, param) {}

		virtual void fill(device_blob &blob) {
			CHECK(blob.count());

			data_type scale = filler_param_.max() - filler_param_.min();
			data_type shift = filler_param_.min();

			ENTER_DEVICE_CONTEXT(blob.device_id())
				CURAND_CHECK(curandGenerateUniform(
					context_->get_device(blob.device_id())->curand_handle(),
					blob.mutable_gpu_data(),
					blob.count()
					));

				cublas_wrapper<data_type>(context_, blob.device_id()).scal_addscalar(
					blob.count(), scale, shift, blob.mutable_gpu_data());
			EXIT_DEVICE_CONTEXT;
		}
	};

	class gaussian_filler : public filler {
	public:
		explicit gaussian_filler(caffepro_context *context, const FillerParameter& param)
			: filler(context, param) {}

		virtual void fill(device_blob &blob) {
			CHECK(blob.count());

			ENTER_DEVICE_CONTEXT(blob.device_id())
				if (blob.count() % 2 == 0) {
					CURAND_CHECK(curandGenerateNormal(
						context_->get_device(blob.device_id())->curand_handle(),
						blob.mutable_gpu_data(),
						blob.count(),
						filler_param_.mean(),
						filler_param_.std()
						));
				}
				else {
					void *buffer = context_->get_current_device()->memory()->allocate((blob.count() + 1) * sizeof(data_type));
					CURAND_CHECK(curandGenerateNormal(
						context_->get_device(blob.device_id())->curand_handle(),
						reinterpret_cast<data_type *>(buffer),
						blob.count() + 1,
						filler_param_.mean(),
						filler_param_.std()
						));
					// leave-one-out copy
					CUDA_CHECK(cudaMemcpyAsync(blob.mutable_gpu_data(), buffer, blob.count() * sizeof(data_type), cudaMemcpyDeviceToDevice));
					context_->get_current_device()->memory()->free(buffer);
				}
			EXIT_DEVICE_CONTEXT;
		}
	};

	class xavier_filler : public filler {
	public:
		explicit xavier_filler(caffepro_context *context, const FillerParameter& param)
			: filler(context, param) {}

		virtual void fill(device_blob &blob) {
			CHECK(blob.count());
			CHECK(blob.is_4d());

			int fan_in = blob.count() / blob.num();
			data_type scale = sqrt(data_type(3) / fan_in) * 2;
			data_type shift = -scale / 2;
			
			ENTER_DEVICE_CONTEXT(blob.device_id())
				CURAND_CHECK(curandGenerateUniform(
					context_->get_device(blob.device_id())->curand_handle(),
					blob.mutable_gpu_data(),
					blob.count()
					));

				cublas_wrapper<data_type>(context_, blob.device_id()).scal_addscalar(
					blob.count(), scale, shift, blob.mutable_gpu_data());
			EXIT_DEVICE_CONTEXT;
		}
	};

	class xiangyu_filler : public filler {
	public:
		explicit xiangyu_filler(caffepro_context *context, const FillerParameter& param)
			: filler(context, param) {}

		virtual void fill(device_blob &blob) {
			CHECK(blob.count());
			CHECK(blob.is_4d());

			int fan_out = blob.count() / blob.channels();
			float gaussian_std = sqrt(2.0f / fan_out);

			ENTER_DEVICE_CONTEXT(blob.device_id())
				if (blob.count() % 2 == 0) {
					CURAND_CHECK(curandGenerateNormal(
						context_->get_device(blob.device_id())->curand_handle(),
						blob.mutable_gpu_data(),
						blob.count(),
						(data_type)0,
						gaussian_std
						));
				}
				else {
					void *buffer = context_->get_current_device()->memory()->allocate((blob.count() + 1) * sizeof(data_type));
					CURAND_CHECK(curandGenerateNormal(
						context_->get_device(blob.device_id())->curand_handle(),
						reinterpret_cast<data_type *>(buffer),
						blob.count() + 1,
						(data_type)0,
						gaussian_std
						));
					// leave-one-out copy
					CUDA_CHECK(cudaMemcpyAsync(blob.mutable_gpu_data(), buffer, blob.count() * sizeof(data_type), cudaMemcpyDeviceToDevice));
					context_->get_current_device()->memory()->free(buffer);
				}
			EXIT_DEVICE_CONTEXT;
		}
	};

	class identity_filler : public filler {
	public:
		explicit identity_filler(caffepro_context *context, const FillerParameter& param)
			: filler(context, param) {}

		virtual void fill(device_blob &blob) {
			data_type* data = blob.write_only_cpu_data();
			CHECK(blob.is_4d());
			
			int n = std::min(blob.num(), blob.channels());

			memset(data, 0, sizeof(data_type)* blob.count());
			for (int i = 0; i < n; i++) {
				data[blob.offset_4d(i, i, blob.height() / 2, blob.width() / 2)] = (data_type)1.f;
			}
		}
	};
}