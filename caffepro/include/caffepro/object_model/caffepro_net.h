
#pragma once 

#include <caffepro/object_model/caffepro_object.h>
#include <caffepro/object_model/node_blob.h>
#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/object_model/caffepro_config.h>
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/context/caffepro_context.h>
#include <caffepro/caffepro.h>
#include <caffepro/proto/caffe.pb.h>

#include <vector>

namespace caffepro {
	class caffepro_net : public caffepro_object {
	public:
		// definations
		struct blob_info {
			std::vector<boost::shared_ptr<node_blob> > blobs, input_blobs, output_blobs;
			std::map<std::string, int> blob_name_to_idx;
		};

		struct layer_runtime_property {
			unsigned int mask_bp_weights, mask_bp_acts;
			unsigned int mask_clear_weights, mask_clear_acts;
			unsigned int fp_release_bottom_mask, bp_release_top_diff_mask, bp_release_top_data_mask;
		};

		struct layer_info {
			std::vector<boost::shared_ptr<caffepro_layer> > layers, data_layers, loss_layers;
			std::map<std::string, int> layer_name_to_idx;
			std::vector<caffepro_layer::layer_io_buffer> bottoms, tops;
			std::vector<layer_runtime_property> runtime_properties;
		};

		struct weight_info {
			std::vector<boost::shared_ptr<node_blob> > weights;
			std::vector<data_type> weight_decay, learning_rate;
		};

		typedef unsigned long long blob_hash;

	protected:
		caffepro_net(caffepro_context *context, const NetParameter &param);
	
	public:
		~caffepro_net();

	public:
		// fetch functions
		const std::string& get_name() const { return name_; }
		void set_name(const std::string &name) { name_ = name; }
		caffepro_context *context() const { return context_; }
		const caffepro_config& config() const { return config_; }
		boost::shared_ptr<data_model::data_provider> data_provider() const { return data_provider_; }

		// blob related
		std::vector<boost::shared_ptr<node_blob> >& blobs() { return blobs_info_.blobs; }
		bool has_blob(const std::string &blob_name) const { return blobs_info_.blob_name_to_idx.count(blob_name) > 0; }
		boost::shared_ptr<node_blob> get_blob(const std::string &blob_name) { 
			CHECK(has_blob(blob_name));
			return blobs_info_.blobs[blobs_info_.blob_name_to_idx[blob_name]];
		}
		std::vector<boost::shared_ptr<node_blob> >& input_blobs() { return blobs_info_.input_blobs; }
		std::vector<boost::shared_ptr<node_blob> >& output_blobs() { return blobs_info_.output_blobs; }

		// layer related
		std::vector<boost::shared_ptr<caffepro_layer> >& layers() { return layers_info_.layers; }
		std::vector<boost::shared_ptr<caffepro_layer> >& data_layers() { return layers_info_.data_layers; }
		std::vector<boost::shared_ptr<caffepro_layer> >& loss_layers() { return layers_info_.loss_layers; }
		bool has_layer(const std::string &layer_name) const { return layers_info_.layer_name_to_idx.count(layer_name) > 0; }
		boost::shared_ptr<caffepro_layer> get_layer(const std::string &layer_name) {
			CHECK(has_layer(layer_name));
			return layers_info_.layers[layers_info_.layer_name_to_idx[layer_name]];
		}
		std::vector<caffepro_layer::layer_io_buffer>& bottoms() { return layers_info_.bottoms; }
		std::vector<caffepro_layer::layer_io_buffer>& tops() { return layers_info_.tops; }
		boost::shared_ptr<node_blob> debug_outputs_fwd() { return debug_outputs_fwd_; }
		boost::shared_ptr<node_blob> debug_outputs_bwd() { return debug_outputs_bwd_; }

		template <typename LayerType>
		static LayerType* get_layer(const std::vector<boost::shared_ptr<caffepro_layer> > &layers) {
			LayerType *result = nullptr;
			for (int i = 0; i < (int)layers.size() && result == nullptr; i++) {
				result = dynamic_cast<LayerType *>(layers[i].get());
			}
			return result;
		}

		// weight related
		std::vector<boost::shared_ptr<node_blob> >& weights() { return weights_info_.weights; }
		weight_info& weights_info() { return weights_info_; }

	protected:
		// utils
		blob_hash get_hash(boost::shared_ptr<node_blob> &blob) { return reinterpret_cast<blob_hash>(blob.get()); }

	public:
		// factory methods
		static caffepro_net *create_from_proto(
			caffepro_context *context, 
			const NetParameter &param, 
			boost::shared_ptr<data_model::data_provider> dataprovider = nullptr
			);

	public:
		// interfaces

		// do not enable save memory when training!!
		void forward(bool save_memory = false, bool fwd_dataprovider = true); 

		// print information about forward to debug outputs
		void forward_debug(bool save_memory = false, bool fwd_dataprovider = true);

		// will NOT FORWARD data provider
		void forward_until(const std::string &last_layer_name, bool save_memory = false);
		
		// will NOT FORWARD data provider
		void forward_range(const std::string &start_layer, const std::string &end_layer, bool save_memory = false);
		
		// forward data provider only, begin the next data prefetch
		void forward_data_provider();

		// you can safely enable save memory options when training
		void backward(bool save_memory = true, bool never_clear_weights = false); 

		// print information about backward to debug outputs
		void backward_debug(bool save_memory = true, bool never_clear_weights = false);

		// call this function when all layers are reshaped
		void finished_reshape();				

		void setup_layer_runtime_properties(bool force_backward_all = false);
		void load_weights(NetParameter &param);
		void load_weights_from_info(NetParameter &param, const analyzer::Info &info);
		void save_proto(NetParameter &param);

		void release_blobs();
		void share_weights_from(caffepro_net &other);

		void mark_as_input_blob(const std::string &blob_name, bool reset_layer_runtime_properties = true);
		void mark_as_output_blob(const std::string &blob_name, bool reset_layer_runtime_properties = true);

		// debug use
		void forward_benchmark(bool save_memory = false);
		void backward_benchmark(bool save_memory = true);

	protected:
		// initialize functions
		void build();
		void init_data_provider(const NetParameter &param);
		bool add_blob(const std::string blob_name, boost::shared_ptr<node_blob> blob);
		void add_layer(const LayerParameter &layer_param, const LayerConnection &layer_connection);
		void setup_input_blobs(const NetParameter &param);
		void setup_output_blobs();
		void setup_weights(const NetParameter &param);
		void check_topology();

	protected:
		caffepro_context *context_;
		NetParameter param_;
		std::string name_;
		caffepro_config config_;
		boost::shared_ptr<data_model::data_provider> data_provider_;

		blob_info blobs_info_;
		layer_info layers_info_;
		weight_info weights_info_;

		// used for network debug
		boost::shared_ptr<node_blob> debug_outputs_fwd_, debug_outputs_bwd_;

	private:
		DISABLE_COPY_AND_ASSIGN(caffepro_net);
	};
}