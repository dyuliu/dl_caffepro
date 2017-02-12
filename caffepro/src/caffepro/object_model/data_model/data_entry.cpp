
#include <caffepro/object_model/data_model/data_entry.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <numeric>

namespace caffepro {
	namespace data_model {
		using std::vector;

		data_entry::data_entry(data_provider &provider, const std::string &name) 
			: provider_(provider), name_(name), enabled_(false) {
			config_.set_config(provider_.config());
			config_.set_default_section_name(name_);
		}

		data_entry::~data_entry() {
			// nothing to do
		}

		void data_entry::init() {
			split_minibatch_ = config_.get_array<int>("split_minibatch", false);
			split_gpu_id_ = config_.get_array<int>("split_gpu_id", false);

			CHECK_EQ(split_minibatch_.size(), split_gpu_id_.size());
		}

		void data_entry::forward() {
			if (!enabled_) return;

			CHECK_EQ(prefetch_buffer_.size(), outputs_.size());
			int num = std::accumulate(split_minibatch_.begin(), split_minibatch_.end(), 0);

			for (int i = 0; i < (int)prefetch_buffer_.size(); i++) {
				if (prefetch_buffer_[i]->size() > 1 || split_minibatch_.size() == 0) { // prefetch buffer provides multi-gpu info
					if (outputs_[i]->size() == 0) { // init output
						outputs_[i]->add_like(provider_.context(), *prefetch_buffer_[i]);
					}
					else {
						CHECK_EQ(outputs_[i]->size(), prefetch_buffer_[i]->size());
						for (int nd = 0; nd < (int)outputs_[i]->size(); nd++) {
							if (prefetch_buffer_[i]->reshaped()) {
								outputs_[i]->get(nd)->reshape_like(*prefetch_buffer_[i]->get(nd));
								prefetch_buffer_[i]->get(nd)->finish_reshape();
							}
						}
					}

					// copy data
					for (int nd = 0; nd < (int)outputs_[i]->size(); nd++) {
						ENTER_DEVICE_CONTEXT(outputs_[i]->get(nd)->device_id())
							CUDA_CHECK(cudaMemcpyAsync(
								outputs_[i]->get(nd)->write_only_gpu_data(),
								prefetch_buffer_[i]->get(nd)->gpu_data_async(),
								prefetch_buffer_[i]->get(nd)->count() * sizeof(data_type),
								cudaMemcpyDeviceToDevice
							));
						EXIT_DEVICE_CONTEXT;
					}
				}
				else { // multi-gpu info provided by config
					CHECK_EQ(split_minibatch_.size(), split_gpu_id_.size());
					CHECK_EQ(prefetch_buffer_[i]->get(0)->num(), num);
					CHECK(prefetch_buffer_[i]->get(0)->get_attr(device_blob::BF_DIM_FIXED_4D));

					if (outputs_[i]->size() == 0) { // init output
						vector<int> dims = prefetch_buffer_[i]->get(0)->dims();
						for (int nd = 0; nd < (int)split_minibatch_.size(); nd++) {
							dims.back() = split_minibatch_[nd];
							outputs_[i]->add(boost::shared_ptr<device_blob>(device_blob::create_nd(
								provider_.context(), (int)dims.size(), &dims[0], split_gpu_id_[nd]
								)));
						}
 					}
					else {
						CHECK_EQ(outputs_[i]->size(), split_minibatch_.size());
						if (prefetch_buffer_[i]->get(0)->reshaped()) {
							vector<int> dims = prefetch_buffer_[i]->get(0)->dims();
							for (int nd = 0; nd < (int)split_minibatch_.size(); nd++) {
								dims.back() = split_minibatch_[nd];
								outputs_[i]->get(nd)->reshape_nd((int)dims.size(), &dims[0]);
							}
							prefetch_buffer_[i]->finish_reshape();
						}
					}

					// copy data
					int start_num = 0;
					for (int nd = 0; nd < (int)outputs_[i]->size(); nd++) {
						ENTER_DEVICE_CONTEXT(outputs_[i]->get(nd)->device_id())
							CUDA_CHECK(cudaMemcpyAsync(
								outputs_[i]->get(nd)->write_only_gpu_data(),
								prefetch_buffer_[i]->get(0)->gpu_data() + prefetch_buffer_[i]->get(0)->offset(start_num), // cannot use gpu_data_async
								outputs_[i]->get(nd)->count() * sizeof(data_type),
								cudaMemcpyDeviceToDevice
							));
						EXIT_DEVICE_CONTEXT;

						start_num += outputs_[i]->get(nd)->num();
					}
				}
			}
		}

		void data_entry::auto_init_buffer(int buffer_index, int num, int channels, int height, int width, bool on_single_device) {
			if (buffer_index >= prefetch_buffer_.size()) {
				prefetch_buffer_.resize(buffer_index + 1);
				outputs_.resize(buffer_index + 1);
			};

			prefetch_buffer_[buffer_index].reset(new node_blob());
			outputs_[buffer_index].reset(new node_blob());
			if (split_minibatch_.size() == 0 || on_single_device) {
				prefetch_buffer_[buffer_index]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
					provider_.context(), num, channels, height, width
					)));
			}
			else {
				int total = std::accumulate(split_minibatch_.begin(), split_minibatch_.end(), 0);
				CHECK_EQ(num, total);

				for (int nd = 0; nd < (int)split_minibatch_.size(); nd++) {
					prefetch_buffer_[buffer_index]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
						provider_.context(), split_minibatch_[nd], channels, height, width, split_gpu_id_[nd]
						)));
				}
			}

			// we do not init outputs_ here because it will be processed in forward 
		}

		void data_entry::write_batch_to_blob(const batch_descriptor &batch, boost::shared_ptr<node_blob> blob, bool fixed_input) {
			auto &batch_data = batch.batch_data;
			CHECK_EQ(batch_data.size(), blob->sum_num());

			bool same_shape = true;
			if (!fixed_input) {
				for (int nd = 0, start = 0; nd < (int)blob->size(); nd++) {
					for (int i = start + 1; i < start + blob->get(nd)->num(); i++) {
						if (batch_data[i].processed_data->data->rows != batch_data[start].processed_data->data->rows
							|| batch_data[i].processed_data->data->cols != batch_data[start].processed_data->data->cols
							|| batch_data[i].processed_data->data->channels() != batch_data[start].processed_data->data->channels()) {
							same_shape = false;
							break;
						}
					}
					if (!same_shape) break;
					start += blob->get(nd)->num();
				}

				if (same_shape) { // same shape per device
					for (int nd = 0, start = 0; nd < (int)blob->size(); nd++) {
						blob->get(nd)->reshape_4d(
							blob->get(nd)->num(), 
							batch_data[start].processed_data->data->channels(),
							batch_data[start].processed_data->data->rows,
							batch_data[start].processed_data->data->cols
							);
						start += blob->get(nd)->num();
					}
				}
			}

			if (!same_shape) { // not fixed size, use ext-dim format
				int img_index = 0;
				for (int nd = 0; nd < (int)blob->size(); nd++) {
					device_blob& dev_blob = *blob->get(nd);
					vector<int> dims(dev_blob.num() * 3);
					vector<data_type> buffer;
					buffer.reserve(dev_blob.count());

					for (int n = 0; n < blob->get(nd)->num(); n++, img_index++) {
						int *dim = &dims[n * 3];
						const cv::Mat &im = *batch_data[img_index].processed_data->data;
						dim[0] = im.cols; // width
						dim[1] = im.rows; // height
						dim[2] = im.channels(); // channels

						int channels = im.channels();
						for (int c = 0; c < channels; ++c) {
							for (int h = 0; h < im.rows; ++h) {
								const float *pPixel = reinterpret_cast<const float *>(im.ptr(h));
								for (int w = 0; w < im.cols; ++w) {
									buffer.push_back(pPixel[w * channels + c]);
								}
							}
						}
					}

					dev_blob.reshape_ext(3, dev_blob.num(), &dims[0]);
					memcpy(dev_blob.mutable_cpu_data(), &buffer[0], buffer.size() * sizeof(data_type));
				}
			}
			else {	// fixed size case
				// pre-resize to avoid multi-thread issue
				for (int nd = 0; nd < (int)blob->size(); nd++) {
					blob->get(nd)->write_only_cpu_data();
				}

#pragma omp parallel for
				for (int img_index = 0; img_index < (int)batch_data.size(); img_index++) {
					int sum = 0, nd = 0;
					while (true) {
						CHECK_LT(nd, blob->size());
						if (sum + blob->get(nd)->num() > img_index) break;
						sum += blob->get(nd)->num();
						nd++;
					}
					int n = img_index - sum;

					const cv::Mat &im = *batch_data[img_index].processed_data->data;
					CHECK_EQ(blob->get(nd)->height(), im.rows);
					CHECK_EQ(blob->get(nd)->width(), im.cols);
					CHECK_EQ(blob->get(nd)->channels(), im.channels());
					int channels = im.channels();
					data_type *target_data = blob->get(nd)->write_only_cpu_data() + blob->get(nd)->offset(n);

					for (int c = 0; c < channels; ++c) {
						for (int h = 0; h < im.rows; ++h) {
							const float *pPixel = reinterpret_cast<const float *>(im.ptr(h));
							for (int w = 0; w < im.cols; ++w) {
								*target_data = pPixel[w * channels + c];
								target_data++;
							}
						}
					}
				}
			}
		}
	}
}