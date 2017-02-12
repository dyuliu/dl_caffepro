
#include <caffepro/object_model/device_blob.h>
#include <caffepro/utils/utils.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/math/reduce_wrapper.h>
#include <caffepro/proto/caffe.pb.h>
#include <memory>

namespace caffepro {

	device_blob::device_blob(caffepro_context *context, int device_id)
		: context_(context), dim_type_(DIMTYPE_UNINITIALIZED), 
		  attr_(0), num_(0), count_(0), ndim_(0), ndim_inner_(0), reshaped_(true), ext_dim_sync_(false) {
		if (device_id < 0) {
			CUDA_CHECK(cudaGetDevice(&device_id));
		}
		device_id_ = device_id;

		memory_manager *mem_manager = nullptr;
		if (context) {
			mem_manager = context->get_device(device_id_)->memory();
		}

		data_.reset(new synced_memory(0, mem_manager, device_id_));
		diff_.reset(new synced_memory(0, mem_manager, device_id_));
		ext_dims_.reset(new synced_memory(0, nullptr, device_id_));
		offsets_.reset(new synced_memory(0, nullptr, device_id_));
	}

	device_blob::~device_blob() {
		// do nothing
	}

	device_blob *device_blob::create_4d(caffepro_context *context, int num, int channels, int height, int width, int device_id) {
		device_blob *blob = new device_blob(context, device_id);
		blob->reshape_4d(num, channels, height, width);
		return blob;
	}

	device_blob *device_blob::create_nd(caffepro_context *context, int ndim, const int dim[], int device_id) {
		device_blob *blob = new device_blob(context, device_id);
		blob->reshape_nd(ndim, dim);
		return blob;
	}

	device_blob *device_blob::create_ext(caffepro_context *context, int ndim_inner, int num, const int dim_arr[], int device_id) {
		device_blob *blob = new device_blob(context, device_id);
		blob->reshape_ext(ndim_inner, num, dim_arr);
		return blob;
	}

	device_blob *device_blob::create_like(caffepro_context *context, const device_blob &other, int device_id) {
		device_blob *blob = new device_blob(context, device_id);
		blob->reshape_like(other);
		return blob;
	}

	device_blob *device_blob::create_like_4dext(caffepro_context *context, const device_blob &other,
		int num_outputs, int k_h, int k_w, int st_h, int st_w, int p_h, int p_w, bool size_floor, int device_id) {
		device_blob *blob = new device_blob(context, device_id);
		blob->reshape_like_4dext(other, num_outputs, k_h, k_w, st_h, st_w, p_h, p_w, size_floor);
		return blob;
	}

	device_blob *device_blob::create_shared(device_blob &other) {
		device_blob *blob = create_like(other.context_, other, other.device_id_);
		blob->share_data(other);
		blob->share_diff(other);
		return blob;
	}

	void device_blob::reshape_4d(int num, int channels, int height, int width) {
		CHECK_GT(num, 0);
		CHECK_GT(channels, 0);
		CHECK_GT(height, 0);
		CHECK_GT(width, 0);

		dim_type_ = DIMTYPE_FIXED_LEN;
		set_attr(BF_DIM_SAME_DIM | BF_DIM_SAME_WIDTH | BF_DIM_SAME_HEIGHT | BF_DIM_SAME_CHANNEL
			| BF_DIM_SAME_COUNT | BF_DIM_FIXED_4D);

		dims_.resize(4);
		dims_[0] = width;
		dims_[1] = height;
		dims_[2] = channels;
		dims_[3] = num;

		num_ = num;
		ndim_inner_ = 3;
		ndim_ = 4;

		int old_count = count_;
		count_ = width * height * channels * num;

		if (count_ != old_count) {
			//CHECK(!get_attr(BF_SHARED_DATA));
			//CHECK(!get_attr(BF_SHARED_DIFF));

			data_->set_size(count_ * sizeof(data_type));
			diff_->set_size(count_ * sizeof(data_type));
		}

		offsets_->set_size(num * sizeof(int));
		int step = count_ / num;
		int *off_data = reinterpret_cast<int *>(offsets_->mutable_cpu_data());
		for (int i = 0; i < num; i++) {
			off_data[i] = i * step;
		}

		reshaped_ = true;
		ext_dim_sync_ = false;
	}

	void device_blob::reshape_nd(int ndim, const int dim[]) {
		CHECK_GT(ndim, 0);

		for (int i = 0; i < ndim; i++) {
			CHECK_GT(dim[i], 0);
		}

		dim_type_ = DIMTYPE_FIXED_LEN;
		set_attr(BF_DIM_SAME_DIM | BF_DIM_SAME_WIDTH | BF_DIM_SAME_HEIGHT | BF_DIM_SAME_CHANNEL | BF_DIM_SAME_COUNT);
		if (ndim == 4) {
			set_attr(BF_DIM_FIXED_4D);
		}
		else {
			clear_attr(BF_DIM_FIXED_4D);
		}

		dims_.clear();
		dims_.insert(dims_.end(), dim, dim + ndim);

		num_ = dims_.back();
		ndim_inner_ = ndim - 1;
		ndim_ = ndim;

		int old_count = count_;
		count_ = 1;
		for (int i = 0; i < ndim; i++) {
			count_ *= dim[i];
		}

		if (count_ != old_count) {
			//CHECK(!get_attr(BF_SHARED_DATA));
			//CHECK(!get_attr(BF_SHARED_DIFF));

			data_->set_size(count_ * sizeof(data_type));
			diff_->set_size(count_ * sizeof(data_type));
		}

		offsets_->set_size(num_ * sizeof(int));
		int step = count_ / num_;
		int *off_data = reinterpret_cast<int *>(offsets_->mutable_cpu_data());
		for (int i = 0; i < num_; i++) {
			off_data[i] = i * step;
		}

		reshaped_ = true;
		ext_dim_sync_ = false;
	}

	void device_blob::reshape_ext(int ndim_inner, int num, const int dim_arr[]) {
		CHECK_GE(ndim_inner, 0);
		CHECK_GT(num, 0);

		int dim_size = num * ndim_inner;
		for (int i = 0; i < dim_size; i++) {
			CHECK_GT(dim_arr[i], 0);
		}

		dim_type_ = DIMTYPE_EXT_LEN;

		num_ = num;
		ndim_inner_ = ndim_inner;
		ndim_ = ndim_inner + 1;

		int old_count = count_;
		offsets_->set_size(num * sizeof(int));
		int *off_data = reinterpret_cast<int *>(offsets_->mutable_cpu_data());

		ext_dims_->set_size(dim_size * sizeof(int));
		int *ext_dims_data = reinterpret_cast<int *>(ext_dims_->mutable_cpu_data());
		memcpy(ext_dims_data, dim_arr, dim_size * sizeof(int));

		unsigned int same_dim = BF_DIM_SAME_DIM;
		unsigned int same_width = BF_DIM_SAME_WIDTH;
		unsigned int same_height = BF_DIM_SAME_HEIGHT;
		unsigned int same_channel = BF_DIM_SAME_CHANNEL;
		unsigned int same_count = BF_DIM_SAME_COUNT;

		off_data[0] = 0;
		int inner_count0 = 0;
		count_ = 0;
		for (int i = 0; i < num; i++) {
			const int *dims = dim_arr + i * ndim_inner;

			int inner_count = 1;
			for (int j = 0; j < ndim_inner; j++) {
				inner_count *= dims[j];

				if (dims[j] != dim_arr[j]) {
					same_dim = 0;
				}
			}

			if (i == 0) {
				inner_count0 = inner_count;
			}
			else if (inner_count0 != inner_count) {
				same_count = 0;
			}

			if (i < num - 1) {
				off_data[i + 1] = off_data[i] + inner_count;
			}
			count_ += inner_count;

			if (ndim_inner != 3 || dims[0] != dim_arr[0]) {
				same_width = 0;
			}
			if (ndim_inner != 3 || dims[1] != dim_arr[1]) {
				same_height = 0;
			}
			if (ndim_inner != 3 || dims[2] != dim_arr[2]) {
				same_channel = 0;
			}
		}

		set_attr(same_dim | same_width | same_height | same_channel | same_count);
		clear_attr(BF_DIM_FIXED_4D);

		if (count_ != old_count) {
			//CHECK(!get_attr(BF_SHARED_DATA));
			//CHECK(!get_attr(BF_SHARED_DIFF));

			data_->set_size(count_ * sizeof(data_type));
			diff_->set_size(count_ * sizeof(data_type));
		}

		reshaped_ = true;
		ext_dim_sync_ = true; // for ext type, ext dim always synced
	}

	void device_blob::reshape_like(const device_blob &other) {
		dim_type_ = other.dim_type_;
		set_attr(other.get_attr(BF_DIM_SAME_DIM | BF_DIM_SAME_WIDTH | BF_DIM_SAME_HEIGHT | BF_DIM_SAME_CHANNEL | BF_DIM_SAME_COUNT | BF_DIM_FIXED_4D));

		dims_ = other.dims_;
		if (other.ext_dims_->size()) {
			ext_dims_->set_size(other.ext_dims_->size());
			memcpy(ext_dims_->mutable_cpu_data(), other.ext_dims_->cpu_data(), other.ext_dims_->size());
		}
		if (other.offsets_->size()) {
			offsets_->set_size(other.offsets_->size());
			memcpy(offsets_->mutable_cpu_data(), other.offsets_->cpu_data(), other.offsets_->size());
		}

		num_ = other.num_;
		int old_count = count_;
		count_ = other.count_;
		ndim_inner_ = other.ndim_inner_;
		ndim_ = other.ndim_;

		if (count_ != old_count) {
			//CHECK(!get_attr(BF_SHARED_DATA));
			//CHECK(!get_attr(BF_SHARED_DIFF));

			data_->set_size(count_ * sizeof(data_type));
			diff_->set_size(count_ * sizeof(data_type));
		}

		reshaped_ = true;
		ext_dim_sync_ = other.ext_dim_sync_;
	}

	void device_blob::reshape_like_4dext(const device_blob &other,
		int num_outputs, int k_h, int k_w, int st_h, int st_w, int p_h, int p_w, bool size_floor) {

		if (other.dim_type_ == DIMTYPE_FIXED_LEN) {
			CHECK(other.get_attr(BF_DIM_FIXED_4D));

			int new_w = calc_output_size(other.width(), k_w, st_w, p_w, size_floor);
			int new_h = calc_output_size(other.height(), k_h, st_h, p_h, size_floor);

			CHECK_GT(new_w, 0);
			CHECK_GT(new_h, 0);

			reshape_4d(other.num_, num_outputs, new_h, new_w);
		}
		else if (other.dim_type_ == DIMTYPE_EXT_LEN) {
			CHECK_EQ(other.ndim(), 4);
			int *new_dims = reinterpret_cast<int *>(alloca(other.num_ * other.ndim_inner_ * sizeof(int)));
			const int *ref_dims = other.ext_dims_cpu();
			for (int i = 0; i < other.num_; i++) {
				const int *src = ref_dims + i * other.ndim_inner_;
				int *dest = new_dims + i * other.ndim_inner_;
				dest[0] = calc_output_size(src[0], k_w, st_w, p_w, size_floor);
				dest[1] = calc_output_size(src[1], k_h, st_h, p_h, size_floor);
				dest[2] = num_outputs;
				CHECK_GT(dest[0], 0);
				CHECK_GT(dest[1], 0);
			}

			reshape_ext(other.ndim_inner_, other.num_, new_dims);
		}

		reshaped_ = true;
		// do not need to set ext_dim_sync_
	}

	void device_blob::share_data(device_blob &other) {
		CHECK_EQ(device_id_, other.device_id_);
		CHECK_LE(count_, other.count_);
		data_ = other.data_;
		set_attr(BF_SHARED_DATA);
		other.set_attr(BF_SHARED_DATA);
	}

	void device_blob::share_diff(device_blob &other) {
		CHECK_EQ(device_id_, other.device_id_);
		CHECK_LE(count_, other.count_);
		diff_ = other.diff_;
		set_attr(BF_SHARED_DIFF);
		other.set_attr(BF_SHARED_DIFF);
	}

	void device_blob::release_data() {
		data_->release_all();
	}

	void device_blob::release_diff() {
		diff_->release_all();
	}

	void device_blob::dev_data_fill_zeros() {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUDA_CHECK(cudaMemset(data_->mutable_gpu_data(), 0, data_->size()));
		EXIT_DEVICE_CONTEXT;
	}
	

	void device_blob::dev_diff_fill_zeros() {
		ENTER_DEVICE_CONTEXT(device_id_)
			CUDA_CHECK(cudaMemset(diff_->mutable_gpu_data(), 0, diff_->size()));
		EXIT_DEVICE_CONTEXT;
	}

	bool device_blob::same_shape(const device_blob &other) const {
		if (dim_type_ != other.dim_type_ || count_ != other.count_ || ndim_ != other.ndim_ || num_ != other.num_) {
			return false;
		}

		if (dim_type_ == DIMTYPE_FIXED_LEN) {
			for (int i = 0; i < ndim_; i++) {
				if (dims_[i] != other.dims_[i]) return false;
			}
		}
		else if (dim_type_ == DIMTYPE_EXT_LEN) {
			const int *dim_data = ext_dims_cpu();
			const int *other_dim_data = other.ext_dims_cpu();
			int dim_size = num_* ndim_inner_;
			for (int i = 0; i < dim_size; i++) {
				if (dim_data[i] != other_dim_data[i]) return false;
			}
		}
		return true;
	}

	bool device_blob::same_dim_at(int d) const {
		CHECK_LT(d, ndim_);
		CHECK_GE(d, 0);

		// for fixed-len dim, always true
		// so we only need to process ext-len dim
		if (dim_type_ == DIMTYPE_EXT_LEN) {
			if (d == ndim_inner_) { // num dim 
				return true;
			}
			else {
				for (int i = 1; i < num_; i++) {
					if (ext_dims_cpu()[i * ndim_inner_ + d] != ext_dims_cpu()[d]) {
						return false;
					}
				}
			}
		}

		return true;
	}

	int device_blob::dim_at(int d) const {
		CHECK(same_dim_at(d));

		if (dim_type_ == DIMTYPE_FIXED_LEN) {
			return dims_[d];
		}
		else if (dim_type_ == DIMTYPE_EXT_LEN) {
			if (d < ndim_inner_) {
				return ext_dims_cpu()[d];
			}
			else {
				return num_;
			}
		}

		return 0;
	}

	void device_blob::fill_data(data_type v) {
		cublas_wrapper<data_type>::fill_constant(device_id_, count_, v, mutable_gpu_data());
	}

	void device_blob::fill_diff(data_type v) {
		cublas_wrapper<data_type>::fill_constant(device_id_, count_, v, mutable_gpu_diff());
	}

	void device_blob::sync_ext_dim() {
		if (!ext_dim_sync_) {
			CHECK_EQ(dim_type_, DIMTYPE_FIXED_LEN);

			int dim_size = num_ * ndim_inner_;
			ext_dims_->set_size(dim_size * sizeof(int));

			int *ext_dim_data = reinterpret_cast<int *>(ext_dims_->mutable_cpu_data());
			for (int i = 0; i < num_; i++) {
				for (int j = 0; j < ndim_inner_; j++) {
					ext_dim_data[j] = dims_[j]; // for fixed-len dim type, dim for all instances should be same
				}
				ext_dim_data += ndim_inner_;
			}

			ext_dim_sync_ = true;
		}
	}

	void device_blob::copy_data_from_via_cpu(const device_blob &other) {
		CHECK_EQ(count_, other.count_);
		
		data_->copy_from_via_cpu(*other.data_);
	}

	void device_blob::copy_data_from_via_gpu(const device_blob &other) {
		CHECK_EQ(count_, other.count_);

		data_->copy_from_via_gpu(*other.data_);
	}

	void device_blob::copy_diff_from_via_cpu(const device_blob &other) {
		CHECK_EQ(count_, other.count_);

		diff_->copy_from_via_cpu(*other.diff_);
	}

	void device_blob::copy_diff_from_via_gpu(const device_blob &other) {
		CHECK_EQ(count_, other.count_);

		diff_->copy_from_via_gpu(*other.diff_);
	}

	void device_blob::load_data_from(BlobProto *proto) {
		CHECK_EQ(count_, proto->width() * proto->height() * proto->channels() * proto->num());
	
		data_type *src_data = proto->mutable_data()->mutable_data();
		data_type *dest_gpu_data = mutable_gpu_data();

		ENTER_DEVICE_CONTEXT(device_id_)
			CUDA_CHECK(cudaMemcpy(dest_gpu_data, src_data, sizeof(data_type)* count_, cudaMemcpyHostToDevice));
		EXIT_DEVICE_CONTEXT;
	}

	void device_blob::save_data_to(BlobProto *proto) {
		if (is_4d()) {
			proto->set_num(num_);
			proto->set_channels(channels());
			proto->set_height(height());
			proto->set_width(width());
		}
		else {
			proto->set_num(count_);
			proto->set_channels(1);
			proto->set_height(1);
			proto->set_width(1);
		}
		proto->clear_data();
		proto->clear_diff();

		const data_type* src_cpu_data = cpu_data();
		for (int i = 0; i < count_; i++) {
			proto->add_data(src_cpu_data[i]);
		}
	}

	void device_blob::save_data(std::ostream &stream) {
		const data_type* src_cpu_data = cpu_data();
		for (int i = 0; i < count_; i++) {
			stream << src_cpu_data[i] << std::endl;
		}
	}

	void device_blob::save_diff(std::ostream &stream) {
		const data_type* src_cpu_diff = cpu_diff();
		for (int i = 0; i < count_; i++) {
			stream << src_cpu_diff[i] << std::endl;
		}
	}

	double device_blob::sum(bool diff) {
		const data_type *data = diff ? gpu_diff() : gpu_data();
		reduce_wrapper reduce(context_, device_id_);
		return reduce.sum(count_, data);
	}

	double device_blob::mean(bool diff) {
		return sum(diff) / count_;
	}

	double device_blob::mean2(bool diff) {
		const data_type *data = diff ? gpu_diff() : gpu_data();
		reduce_wrapper reduce(context_, device_id_);
		return reduce.sum2(count_, data) / count_;
	}

	double device_blob::variance(bool diff) {
		double m = mean(diff);
		return std::max(mean2(diff) - m * m, 0.);
	}

	data_type device_blob::max(bool diff) {
		const data_type *data = diff ? gpu_diff() : gpu_data();

		reduce_wrapper reduce(context_, device_id_);
		return reduce.max(count_, data);
	}

	data_type device_blob::min(bool diff) {
		const data_type *data = diff ? gpu_diff() : gpu_data();

		reduce_wrapper reduce(context_, device_id_);
		return reduce.min(count_, data);
	}
}