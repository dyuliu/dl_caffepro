
#pragma once

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/context/caffepro_context.h>
#include <caffepro/context/memory/syncd_memory.h>
#include <caffepro/caffepro.h>

#include <boost/shared_ptr.hpp>
#include <vector>
#include <fstream>

namespace caffepro {

	class BlobProto;

	class device_blob : public caffepro_object {

	public:
		// definations
		enum blob_dim_type {
			DIMTYPE_UNINITIALIZED,
			DIMTYPE_FIXED_LEN,
			DIMTYPE_EXT_LEN
		};

		enum blob_flags : unsigned int {
			BF_DIM_SAME_DIM			= 1U << 0,
			BF_DIM_SAME_WIDTH		= 1U << 1,
			BF_DIM_SAME_HEIGHT		= 1U << 2,
			BF_DIM_SAME_CHANNEL		= 1U << 3,
			BF_DIM_SAME_COUNT		= 1U << 4,
			BF_DIM_FIXED_4D			= 1U << 5,
			BF_SHARED_DATA			= 1U << 6,
			BF_SHARED_DIFF			= 1U << 7
		};

		typedef unsigned long long blob_hash;

	protected:
		device_blob(caffepro_context *context, int device_id);

	public:
		~device_blob();

	public:
		// factory methods
		static device_blob *create_4d(caffepro_context *context, int num, int channels, int height, int width, int device_id = -1);
		static device_blob *create_nd(caffepro_context *context, int ndim, const int dim[], int device_id = -1);
		static device_blob *create_ext(caffepro_context *context, int ndim_inner, int num, const int dim_arr[], int device_id = -1);
		static device_blob *create_like(caffepro_context *context, const device_blob &other, int device_id = -1);
		static device_blob *create_like_4dext(caffepro_context *context, const device_blob &other, 
			int num_outputs, int k_h, int k_w, int st_h, int st_w, int p_h, int p_w, bool size_floor, int device_id = -1);
		static device_blob *create_shared(device_blob &other);

	public:
		// reshape
		void reshape_4d(int num, int channels, int height, int width);
		void reshape_nd(int ndim, const int dim[]);
		void reshape_ext(int ndim_inner, int num, const int dim_arr[]);
		void reshape_like(const device_blob &other);
		void reshape_like_4dext(const device_blob &other, int num_outputs, int k_h, int k_w, int st_h, int st_w, int p_h, int p_w, bool size_floor);

		// share
		void share_data(device_blob &other);
		void share_diff(device_blob &other);

		// other
		void release_data();
		void release_diff();
		void dev_data_fill_zeros();
		void dev_diff_fill_zeros();
		bool same_shape(const device_blob &other) const;
		bool same_dim_at(int d) const;
		int dim_at(int d) const;
		void fill_data(data_type v);
		void fill_diff(data_type v);
		void sync_ext_dim();

		// data
		void copy_data_from_via_cpu(const device_blob &other);
		void copy_data_from_via_gpu(const device_blob &other);
		void copy_diff_from_via_cpu(const device_blob &other);
		void copy_diff_from_via_gpu(const device_blob &other);

		void load_data_from(BlobProto *proto);
		void save_data_to(BlobProto *proto);

		void save_data(std::ostream &stream);
		void save_diff(std::ostream &stream);

		// statistic
		double sum(bool diff = false);
		double mean(bool diff = false);
		double mean2(bool diff = false);
		double variance(bool diff = false);
		data_type max(bool diff = false);
		data_type min(bool diff = false);

	public:
		// fetch functions
		unsigned int get_attr(unsigned int mask) const { return attr_ & mask; }
		void set_attr(unsigned int mask) { attr_ |= mask; }
		void clear_attr(unsigned int mask) { attr_ &= ~mask; }
		blob_hash get_data_hash() const { return reinterpret_cast<blob_hash>(data_.get()); }
		blob_hash get_diff_hash() const { return reinterpret_cast<blob_hash>(diff_.get()); }
		caffepro_context* context() const { return context_; }
		int device_id() const { return device_id_; }
		int reshaped() const { return reshaped_; }
		void finish_reshape() { reshaped_ = false; }

		// dim related
		blob_dim_type dim_type() const { return dim_type_; }
		bool is_4d() const { 
			return dim_type_ == DIMTYPE_FIXED_LEN && get_attr(BF_DIM_FIXED_4D)
				|| dim_type_ == DIMTYPE_EXT_LEN && ndim_ == 4 && get_attr(BF_DIM_SAME_DIM);
		}
		int width() const { CHECK(is_4d()); return dim_type_ == DIMTYPE_FIXED_LEN ? dims_[0] : ext_dims_cpu()[0]; }
		int height() const { CHECK(is_4d()); return dim_type_ == DIMTYPE_FIXED_LEN ? dims_[1] : ext_dims_cpu()[1]; }
		int channels() const { CHECK(is_4d()); return dim_type_ == DIMTYPE_FIXED_LEN ? dims_[2] : ext_dims_cpu()[2]; }
		int num() const { return num_; }
		int count() const { return count_; }
		int inner_count() const { CHECK(get_attr(BF_DIM_SAME_COUNT)); return count_ / num_; }
		const std::vector<int> &dims() const { CHECK_EQ(dim_type_, DIMTYPE_FIXED_LEN); return dims_; }
		int ndim() const { return ndim_; }
		int ndim_inner() const { return ndim_inner_; }
		const int *ext_dims_cpu() const { CHECK(ext_dim_sync_); return reinterpret_cast<const int *>(ext_dims_->cpu_data()); }
		const int *ext_dims_gpu() const { CHECK(ext_dim_sync_); return reinterpret_cast<const int *>(ext_dims_->gpu_data()); }
		const int *offsets_cpu() const { return reinterpret_cast<const int *>(offsets_->cpu_data()); }
		const int *offsets_gpu() const { return reinterpret_cast<const int *>(offsets_->gpu_data()); }
		int offset(int num) { return offsets_cpu()[num]; }
		int offset_4d(int n = 0, int c = 0, int h = 0, int w = 0) {
			CHECK(is_4d());
			return ((n * channels() + c) * height() + h) * width() + w;
		}

		// data related
		const data_type* gpu_data() const { return reinterpret_cast<const data_type *>(data_->gpu_data()); }
		const data_type* cpu_data() const { return reinterpret_cast<const data_type *>(data_->cpu_data()); }
		const data_type* gpu_data_async() const { return reinterpret_cast<const data_type *>(data_->async_gpu_data()); }
		data_type* mutable_gpu_data() { return reinterpret_cast<data_type *>(data_->mutable_gpu_data()); }
		data_type* mutable_cpu_data() { return reinterpret_cast<data_type *>(data_->mutable_cpu_data()); }
		data_type* write_only_gpu_data() { return reinterpret_cast<data_type *>(data_->write_only_gpu_data()); }
		data_type* write_only_cpu_data() { return reinterpret_cast<data_type *>(data_->write_only_cpu_data()); }
		const data_type* gpu_diff() const { return reinterpret_cast<const data_type *>(diff_->gpu_data()); }
		const data_type* cpu_diff() const { return reinterpret_cast<const data_type *>(diff_->cpu_data()); }
		const data_type* gpu_diff_async() const { return reinterpret_cast<const data_type *>(diff_->async_gpu_data()); }
		data_type* mutable_gpu_diff() { return reinterpret_cast<data_type *>(diff_->mutable_gpu_data()); }
		data_type* mutable_cpu_diff() { return reinterpret_cast<data_type *>(diff_->mutable_cpu_data()); }
		data_type* write_only_gpu_diff() { return reinterpret_cast<data_type *>(diff_->write_only_gpu_data()); }
		data_type* write_only_cpu_diff() { return reinterpret_cast<data_type *>(diff_->write_only_cpu_data()); }

		boost::shared_ptr<synced_memory> data_storage() const { return data_; }
		boost::shared_ptr<synced_memory> diff_storage() const { return diff_; }

	protected:
		// commom members
		caffepro_context *context_;
		boost::shared_ptr<synced_memory> data_, diff_;
		int device_id_;
		blob_dim_type dim_type_;
		unsigned int attr_;
		bool reshaped_;
		bool ext_dim_sync_;

		// dimensions
		std::vector<int> dims_; // for fixed-len dim
		boost::shared_ptr<synced_memory> ext_dims_; // for ext-len dim
		boost::shared_ptr<synced_memory> offsets_;
		int num_, count_, ndim_inner_, ndim_;

	private:
		DISABLE_COPY_AND_ASSIGN(device_blob);
	};
}