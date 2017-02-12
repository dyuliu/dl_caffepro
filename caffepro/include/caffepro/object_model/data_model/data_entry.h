
#pragma once 

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/object_model/caffepro_layer.h>
#include <opencv2/opencv.hpp>

namespace caffepro {
	namespace data_model {
		class data_provider;
		struct batch_descriptor;

		class data_entry : public caffepro_object {
		public:
			data_entry(data_provider &provider, const std::string &name);
			virtual ~data_entry();

		public:
			// fetch functions
			data_provider &provider() const { return provider_; }
			const std::string &name() const { return name_; }
			const std::string &type() const { return type_; }
			const std::vector<std::string> &required_entries() const { return required_entries_; }
			bool get_enable_state() const { return enabled_; }
			void set_enabled(bool enable) { enabled_ = enable; }

			caffepro_layer::layer_io_buffer &prefetch_buffer() { return prefetch_buffer_; }
			const caffepro_layer::layer_io_buffer &prefetch_buffer() const { return prefetch_buffer_; }
			caffepro_layer::layer_io_buffer &outputs() { return outputs_; }
			const caffepro_layer::layer_io_buffer &outputs() const { return outputs_; }

		public:
			virtual void init();
			virtual void prepare(batch_descriptor &batch) = 0;
			virtual void forward();

		protected:
			// util functions
			void auto_init_buffer(int buffer_index, int num, int channels, int height, int width, bool on_single_device);
			void write_batch_to_blob(const batch_descriptor &batch, boost::shared_ptr<node_blob> blob, bool fixed_input);

		protected:
			// common
			data_provider &provider_;
			std::string name_, type_;
			std::vector<std::string> required_entries_;
			caffepro_config_reader config_;
			bool enabled_;

			std::vector<int> split_minibatch_, split_gpu_id_;

			// output
			caffepro_layer::layer_io_buffer prefetch_buffer_, outputs_;
		};
	}
}