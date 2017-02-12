
#include <caffepro/object_model/node_blob.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	node_blob::node_blob()
		: attr_(0) {
	}

	node_blob::~node_blob() {
		// nothing to do
	}

	void node_blob::add(boost::shared_ptr<device_blob> blob) {
		dev_blobs_.push_back(blob);
	}

	void node_blob::add_like(const device_blob &other) {
		dev_blobs_.push_back(boost::shared_ptr<device_blob>(
			device_blob::create_like(other.context(), other, other.device_id())));
	}

	void node_blob::add_like(caffepro_context *context, const device_blob &other) {
		dev_blobs_.push_back(boost::shared_ptr<device_blob>(
			device_blob::create_like(context, other, other.device_id())));
	}

	void node_blob::add_like(const node_blob &other) {
		for (int i = 0; i < (int)other.size(); i++) {
			add_like(*other[i]);
		}
	}

	void node_blob::add_like(caffepro_context *context, const node_blob &other) {
		for (int i = 0; i < (int)other.size(); i++) {
			add_like(context, *other[i]);
		}
	}

	void node_blob::add_like(const device_blob &shape, const node_blob &device_sources) {
		CHECK_GT(device_sources.size(), 0);

		for (int i = 0; i < (int)device_sources.size(); i++) {
			dev_blobs_.push_back(boost::shared_ptr<device_blob>(
				device_blob::create_like(device_sources[i]->context(), shape, device_sources[i]->device_id())));
		}
	}
	void node_blob::add_like(caffepro_context *context, const device_blob &shape, const node_blob &device_sources) {
		CHECK_GT(device_sources.size(), 0);

		for (int i = 0; i < (int)device_sources.size(); i++) {
			dev_blobs_.push_back(boost::shared_ptr<device_blob>(
				device_blob::create_like(context, shape, device_sources[i]->device_id())));
		}
	}

	void node_blob::set_4d(int device_index, int num, int channels, int height, int width, int device_id, caffepro_context *context) {
		if (device_index >= (int)dev_blobs_.size()) {
			dev_blobs_.resize(device_index + 1);
		}

		if (!dev_blobs_[device_index]) {
			dev_blobs_[device_index].reset(device_blob::create_4d(context, num, channels, height, width, device_id));
		}
		else {
			if (context) {
				CHECK_EQ(dev_blobs_[device_index]->context(), context);
			}
			if (device_id >= 0) {
				CHECK_EQ(dev_blobs_[device_index]->device_id(), device_id);
			}

			dev_blobs_[device_index]->reshape_4d(num, channels, height, width);
		}
	}

	void node_blob::bind_as_layer_input(caffepro_layer *layer) {
		input_bindings_.push_back(layer);
	}

	void node_blob::bind_as_layer_output(caffepro_layer *layer) {
		output_bindings_.push_back(layer);
	}

	bool node_blob::same_dimtype() const {
		for (int i = 1; i < (int)dev_blobs_.size(); i++) {
			if (dev_blobs_[i]->dim_type() != dev_blobs_[0]->dim_type()) {
				return false;
			}
		}

		return true;
	}

	bool node_blob::same_ndim() const {
		for (int i = 1; i < (int)dev_blobs_.size(); i++) {
			if (dev_blobs_[i]->ndim() != dev_blobs_[0]->ndim()) {
				return false;
			}
		}

		return true;
	}

	bool node_blob::same_count() const {
		for (int i = 1; i < (int)dev_blobs_.size(); i++) {
			if (dev_blobs_[i]->count() != dev_blobs_[0]->count()) {
				return false;
			}
		}

		return true;
	}

	bool node_blob::same_num() const {
		for (int i = 1; i < (int)dev_blobs_.size(); i++) {
			if (dev_blobs_[i]->num() != dev_blobs_[0]->num()) {
				return false;
			}
		}

		return true;
	}

	bool node_blob::same_dim_at(int d) const {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (!dev_blobs_[i]->same_dim_at(d)) return false;

			if (i > 0) {
				if (dev_blobs_[i]->dim_at(d) != dev_blobs_[0]->dim_at(d)) {
					return false;
				}
			}
		}

		return true;
	}

	bool node_blob::same_inner_count() const {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (!dev_blobs_[i]->get_attr(device_blob::BF_DIM_SAME_COUNT)) return false;

			if (i > 0) {
				if (dev_blobs_[i]->inner_count() != dev_blobs_[0]->inner_count()) {
					return false;
				}
			}
		}

		return true;
	}

	int node_blob::dim_at(int d) const {
		CHECK_GT(dev_blobs_.size(), 0);
		CHECK(same_dim_at(d));

		return dev_blobs_[0]->dim_at(d);
	}

	bool node_blob::reshaped() const {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (dev_blobs_[i]->reshaped()) return true;
		}
		return false;
	}

	int node_blob::sum_num() const {
		int num = 0;
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			num += dev_blobs_[i]->num();
		}
		return num;
	}

	void node_blob::finish_reshape() {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			dev_blobs_[i]->finish_reshape();
		}
	}

	void node_blob::broadcast_data_via_cpu(int src_device_index) {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (i != src_device_index) {
				dev_blobs_[i]->copy_data_from_via_cpu(*get(src_device_index));
			}
		}
	}

	void node_blob::broadcast_data_via_gpu(int src_device_index) {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (i != src_device_index) {
				dev_blobs_[i]->copy_data_from_via_gpu(*get(src_device_index));
			}
		}
	}

	void node_blob::broadcast_diff_via_cpu(int src_device_index) {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (i != src_device_index) {
				dev_blobs_[i]->copy_diff_from_via_cpu(*get(src_device_index));
			}
		}
	}

	void node_blob::broadcast_diff_via_gpu(int src_device_index) {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			if (i != src_device_index) {
				dev_blobs_[i]->copy_diff_from_via_gpu(*get(src_device_index));
			}
		}
	}

	void node_blob::release_data() {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			dev_blobs_[i]->release_data();
		}
	}

	void node_blob::release_diff() {
		for (int i = 0; i < (int)dev_blobs_.size(); i++) {
			dev_blobs_[i]->release_diff();
		}
	}

	void node_blob::load_data_from(BlobProto *proto) {
		CHECK_GT(dev_blobs_.size(), 0);

		dev_blobs_[0]->load_data_from(proto);
		broadcast_data_via_gpu(0);
	}

	void node_blob::save_data_to(BlobProto *proto) {
		CHECK_GT(dev_blobs_.size(), 0);

		dev_blobs_[0]->save_data_to(proto);
	}

	void node_blob::offset_across_dev(int n, __out int *device_id, __out int *off) const {
		CHECK_LT(n, sum_num());

		for (int nd = 0; nd < (int)dev_blobs_.size(); nd++) {
			if (n < dev_blobs_[nd]->num()) {
				if (device_id) {
					*device_id = nd;
				}
				if (off) {
					*off = dev_blobs_[nd]->offset(n);
				}
				break;
			}
			else {
				n -= dev_blobs_[nd]->num();
			}
		}
	}
}