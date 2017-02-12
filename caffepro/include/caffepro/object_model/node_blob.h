
#pragma once 

#include <caffepro/object_model/device_blob.h>
#include <string>

namespace caffepro {

	class caffepro_layer;
	class BlobProto;

	class node_blob : public caffepro_object {
	public:
		// definations 
		enum node_flags : unsigned int {
			NF_NET_INPUT					= 1 << 0,
			NF_NET_OUTPUT					= 1 << 1,
			NF_TEMP							= 1 << 2,
			NF_BIND_INPLACE					= 1 << 3,
			NF_BIND_FORBID_INPLACE_USAGE	= 1 << 4
		};

	public:
		node_blob();
		~node_blob();

	public:
		// interfaces
		void add(boost::shared_ptr<device_blob> blob);
		void add_like(const device_blob &other);
		void add_like(caffepro_context *context, const device_blob &other);
		void add_like(const node_blob &other);
		void add_like(caffepro_context *context, const node_blob &other);
		void add_like(const device_blob &shape, const node_blob &device_sources);
		void add_like(caffepro_context *context, const device_blob &shape, const node_blob &device_sources);

		void set_4d(int device_index, int num, int channels, int height, int width, int device_id = -1, caffepro_context *context = nullptr);

		void bind_as_layer_input(caffepro_layer *layer);
		void bind_as_layer_output(caffepro_layer *layer);

		bool same_dimtype() const;
		bool same_ndim() const;
		bool same_count() const;
		bool same_num() const;
		bool same_dim_at(int d) const;
		bool same_inner_count() const;
		int dim_at(int d) const;
		bool reshaped() const;
		int sum_num() const;
		void finish_reshape();

		void broadcast_data_via_cpu(int src_device_index);
		void broadcast_data_via_gpu(int src_device_index);
		void broadcast_diff_via_cpu(int src_device_index);
		void broadcast_diff_via_gpu(int src_device_index);

		void release_data();
		void release_diff();

		void load_data_from(BlobProto *proto);
		void save_data_to(BlobProto *proto);

		void offset_across_dev(int n, __out int *device_id = nullptr, __out int *off = nullptr) const;

	public:
		// fetch functions
		boost::shared_ptr<device_blob> get(int device_index) { 
			CHECK_GE(device_index, 0);
			CHECK_LT(device_index, dev_blobs_.size());
			return dev_blobs_[device_index];
		}

		const boost::shared_ptr<device_blob> get(int device_index) const {
			CHECK_GE(device_index, 0);
			CHECK_LT(device_index, dev_blobs_.size());
			return dev_blobs_[device_index];
		}

		boost::shared_ptr<device_blob> operator[] (int device_index) {
			return get(device_index);
		}

		const boost::shared_ptr<device_blob> operator[] (int device_index) const {
			return get(device_index);
		}

		size_t size() const { return dev_blobs_.size(); }
		void clear() { dev_blobs_.clear(); }
		std::vector<std::string> &tags() { return tags_; }
		unsigned int get_attr(unsigned int mask) const { return attr_ & mask; }
		void set_attr(unsigned int mask) { attr_ |= mask; }
		void clear_attr(unsigned int mask) { attr_ &= ~mask; }
		std::vector<caffepro_layer *> &input_bindings() { return input_bindings_; }
		std::vector<caffepro_layer *> &output_bindings() { return output_bindings_; }
		void clear_input_bindings() { input_bindings_.clear(); }
		void clear_output_bindings() { output_bindings_.clear(); }
		const std::string &get_name() const { return name_; }
		void set_name(const std::string &name) { name_ = name; }

		const data_type* get_gpu_data_across_dev(int n) const { 
			int device_id = -1, offset = 0;
			offset_across_dev(n, &device_id, &offset);
			return dev_blobs_[device_id]->gpu_data() + dev_blobs_[device_id]->offset(offset);
		}

		const data_type* get_cpu_data_across_dev(int n) const { 
			int device_id = -1, offset = 0;
			offset_across_dev(n, &device_id, &offset);
			return dev_blobs_[device_id]->cpu_data() + offset;
		}

	protected:
		// members
		std::vector<boost::shared_ptr<device_blob> > dev_blobs_;
		std::vector<std::string> tags_;
		std::vector<caffepro_layer *> input_bindings_, output_bindings_;
		std::string name_;
		unsigned int attr_;

	private:
		DISABLE_COPY_AND_ASSIGN(node_blob);
	};
}
