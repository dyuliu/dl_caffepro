
#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/utils/utils.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/color_print.h>
#include <algorithm>
#include <sstream>
#include <set>

using std::string;
using std::vector;
using std::set;
using std::map;

namespace caffepro {
	caffepro_net::caffepro_net(caffepro_context *context, const NetParameter &param)
		: context_(context), param_(param) {
		CHECK(context);
	}

	caffepro_net::~caffepro_net() {
		// nothing to do
	}

	caffepro_net *caffepro_net::create_from_proto(caffepro_context *context, const NetParameter &param, boost::shared_ptr<data_model::data_provider> dataprovider) {
		caffepro_net *net = new caffepro_net(context, param);
		net->data_provider_ = dataprovider;
		net->build();
		return net;
	}

	void caffepro_net::build() {
		name_ = param_.name();

		// load config
		if (param_.has_config_file()) {
			config_.load_config(param_.config_file());
		}

		// init data_provider
		init_data_provider(param_);
		
		blobs_info_.blobs.clear();
		blobs_info_.blob_name_to_idx.clear();

		// first, set input blobs
		setup_input_blobs(param_);

		// second, set up layers
		layers_info_.layers.clear();
		layers_info_.layer_name_to_idx.clear();
		layers_info_.bottoms.clear();
		layers_info_.tops.clear();
		layers_info_.runtime_properties.clear();

		for (int i = 0; i < (int)param_.layers_size(); i++) {
			const LayerConnection& layer_connection = param_.layers(i);
			const LayerParameter& layer_param = layer_connection.layer();

			add_layer(layer_param, layer_connection);
		}

		// third, set up output blobs
		setup_output_blobs();

		// fourth, set up layer runtime properties
		setup_layer_runtime_properties();
		
		// fifth, set up weights
		setup_weights(param_);

		// final check the topology
		check_topology();
		finished_reshape();

		LOG(INFO) << "Network initialization done.";
	}

	void caffepro_net::init_data_provider(const NetParameter &param) {
		if (!data_provider_ && param.has_data_provider_name()) {
			data_provider_.reset(new data_model::data_provider(context_, &config_));
			data_provider_->set_data_provider_name(param.data_provider_name());
			data_provider_->auto_build();
		}

		if (data_provider_) {
			data_provider_->set_namespace(name_);
			data_provider_->init(); // including prefetch
			data_provider_->forward();
		}
	}

	bool caffepro_net::add_blob(const std::string blob_name, boost::shared_ptr<node_blob> blob) {
		if (has_blob(blob_name)) {
			return false;
		}

		blobs_info_.blobs.push_back(blob);
		blobs_info_.blob_name_to_idx[blob_name] = (int)blobs_info_.blobs.size() - 1;
		blob->set_name(blob_name);
		return true;
	}

	void caffepro_net::setup_input_blobs(const NetParameter &param) {
		blobs_info_.input_blobs.clear();
		CHECK_EQ(param.input_size() * 4, param.input_dim_size()) << "Incorrect bottom blob dimension specifications.";

		for (int i = 0; i < (int)param.input_size(); ++i) {
			const string& blob_name = param.input(i);
			boost::shared_ptr<node_blob> blob(new node_blob());
			blob->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
					context_,
					param.input_dim(i * 4),
					param.input_dim(i * 4 + 1),
					param.input_dim(i * 4 + 2),
					param.input_dim(i * 4 + 3),
					-1
					)
				)); // create on current gpu

			blob->set_attr(node_blob::NF_NET_INPUT);
			CHECK(add_blob(blob_name, blob)) << "Duplicate input blob name";
			blobs_info_.input_blobs.push_back(blob);
		}
	}

	void caffepro_net::add_layer(const LayerParameter &layer_param, const LayerConnection &layer_connection) {
		LOG(INFO) << "Creating Layer " << layer_param.name();
		boost::shared_ptr<caffepro_layer> layer(caffepro_layer::create(context_, layer_param));
		CHECK(!has_layer(layer_param.name())) << "Duplicate layer name";

		layer->config().set_config(&config_);
		layer->config().set_default_section_name(layer_param.name());
		layer->set_namespace(name_);

		layers_info_.layers.push_back(layer);
		layers_info_.layer_name_to_idx[layer_param.name()] = (int)layers_info_.layers.size() - 1;
	
		if (layer->attr().usage == caffepro_layer::layer_attribute::USAGE_DATA_SOURCE) {
			layers_info_.data_layers.push_back(layer);
		}
		else if (layer->attr().usage == caffepro_layer::layer_attribute::USAGE_LOSS) {
			layers_info_.loss_layers.push_back(layer);
		}

		caffepro_layer::layer_io_buffer bottom, top;

		// add connections
		for (int j = 0; j < (int)layer_connection.bottom_size(); j++) {
			const string& blob_name = layer_connection.bottom(j);
			CHECK(has_blob(blob_name)) << "Unknown blob input " << blob_name << " in layer " << layer_param.name();

			bottom.push_back(get_blob(blob_name));
		}

		for (int j = 0; j < (int)layer_connection.top_size(); j++) {
			const string& blob_name = layer_connection.top(j);

			if (has_blob(blob_name)) {
				if (j > 0 || bottom.size() == 0 || layer_connection.bottom(0) != blob_name) {
					LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
				}
				else {
					LOG(INFO) << layer_param.name() << " -> " << blob_name << " (in-place)";
				}
				top.push_back(get_blob(blob_name));
			}
			else {
				LOG(INFO) << layer_param.name() << " -> " << blob_name;
				top.push_back(boost::shared_ptr<node_blob>(new node_blob()));
				add_blob(blob_name, top.back());
			}
		}

		// init layer
		layers_info_.bottoms.push_back(bottom);
		layers_info_.tops.push_back(top);
		layer->bind(bottom, top);
		layer->init();
		layer->resize();

		// process shared weights
		if (layer_param.has_share_weights()) {
			const string share_layer_name = layer_param.share_weights();
			CHECK(has_layer(share_layer_name)) << "Layer " << share_layer_name << " to share with does not exist";
			boost::shared_ptr<caffepro_layer> src_layer = get_layer(share_layer_name);

			CHECK_EQ(src_layer->weights().size(), layer->weights().size())
				<< "Different number of weights for shared layer: " << share_layer_name;

			for (int i = 0; i < (int)layer->weights().size(); i++) {
				boost::shared_ptr<node_blob> current_weight = layer->weights()[i];
				boost::shared_ptr<node_blob> src_weight = src_layer->weights()[i];

				CHECK_EQ(current_weight->size(), src_weight->size())
					<< "Different number of devices for shared layer: " << share_layer_name;
				for (int nd = 0; nd < (int)src_weight->size(); nd++) {
					CHECK_EQ(current_weight->get(nd)->count(), src_weight->get(nd)->count())
						<< "Different size of weights for shared layer: " << share_layer_name;
					CHECK_EQ(current_weight->get(nd)->device_id(), src_weight->get(nd)->device_id());
				}

				// set share
				layer->weights()[i] = src_layer->weights()[i];
			}
		}

		// print top shape
		for (int topid = 0; topid < top.size(); ++topid) {
			for (int nd = 0; nd < top[topid]->size(); nd++) {
				std::stringstream info;
				info << "Top shape: (GPU: " << top[topid]->get(nd)->device_id() << ") ";
				if (top[topid]->get(nd)->dim_type() == device_blob::DIMTYPE_FIXED_LEN) {
					for (int i = 0; i < top[topid]->get(nd)->ndim(); i++) {
						info << top[topid]->get(nd)->dims()[i] << " ";
					}
				}
				else if (top[topid]->get(nd)->dim_type() == device_blob::DIMTYPE_EXT_LEN) {
					info << "ext dim " << top[topid]->get(nd)->ndim() << " ";
				}
				info << "(" << top[topid]->get(nd)->count() << ")";
				LOG(INFO) << info.str();
			}
		}
	}

	void caffepro_net::setup_output_blobs() {
		blobs_info_.output_blobs.clear();

		set<blob_hash> visible_blobs;
		for (int i = 0; i < (int)blobs_info_.input_blobs.size(); i++) {
			visible_blobs.insert(get_hash(blobs_info_.input_blobs[i]));
		}
		for (int i = 0; i < (int)layers_info_.layers.size(); i++) {
			for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
				visible_blobs.erase(get_hash(layers_info_.bottoms[i][j]));
			}
			for (int j = 0; j < (int)layers_info_.tops[i].size(); j++) {
				visible_blobs.insert(get_hash(layers_info_.tops[i][j]));
			}
		}

		// output blobs are the visible blobs after all layers
		for (int i = 0; i < (int)blobs_info_.blobs.size(); i++) {
			if (visible_blobs.count(get_hash(blobs_info_.blobs[i]))) {
				blobs_info_.blobs[i]->set_attr(node_blob::NF_NET_OUTPUT);
				blobs_info_.output_blobs.push_back(blobs_info_.blobs[i]);
				LOG(INFO) << "Output blob detected: " << blobs_info_.blobs[i]->get_name();
			}
		}
	}

	void caffepro_net::setup_layer_runtime_properties(bool force_backward_all) {
		int n_layers = (int)layers_info_.layers.size();
		layers_info_.runtime_properties.resize(n_layers);
		memset(&layers_info_.runtime_properties[0], 0, sizeof(layer_runtime_property)* n_layers);

		// generate bp weights and acts masks
		set<blob_hash> need_bp_blob_hash;
		for (int i = 0; i < n_layers; i++) {
			unsigned int mask_bp_acts = 0;

			for (int j = 0; j < (int)layers_info_.layers[i]->inputs().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->inputs()[j]);
				if (force_backward_all || need_bp_blob_hash.count(h)) {
					mask_bp_acts |= 1U << j;
				}
			}
			layers_info_.runtime_properties[i].mask_bp_acts = mask_bp_acts;

			unsigned int mask_bp_weights = 0;
			auto &layer_param = layers_info_.layers[i]->layer_param();
			for (int j = 0; j < (int)layers_info_.layers[i]->weights().size(); j++) {

				if (j >= layer_param.blobs_lr_size() || layer_param.blobs_lr(j) != 0) {
					mask_bp_weights |= 1U << j;
				}
			}
			layers_info_.runtime_properties[i].mask_bp_weights = mask_bp_weights;

			if (mask_bp_acts || mask_bp_weights) {
				for (int j = 0; j < (int)layers_info_.layers[i]->outputs().size(); j++) {
					blob_hash h = get_hash(layers_info_.layers[i]->outputs()[j]);
					need_bp_blob_hash.insert(h);
				}
			}
		}

		// generate fp release masks
		map<blob_hash, int> bottom_ref_count;
		for (int i = 0; i < n_layers; i++) {
			for (int j = 0; j < (int)layers_info_.layers[i]->inputs().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->inputs()[j]);
				bottom_ref_count[h]++;
			}
		}
		for (int i = 0; i < n_layers; i++) {
			unsigned int fp_release_bottom_mask = 0;
			for (int j = 0; j < (int)layers_info_.layers[i]->inputs().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->inputs()[j]);
				bottom_ref_count[h]--;
				if (bottom_ref_count[h] == 0) {
					if (!layers_info_.layers[i]->inputs()[j]->get_attr(node_blob::NF_NET_OUTPUT)) {
						fp_release_bottom_mask |= 1U << j;
					}
				}
			}
			layers_info_.runtime_properties[i].fp_release_bottom_mask = fp_release_bottom_mask;
		}

		// generate bp release mask
		map<blob_hash, int> top_ref_count;
		for (int i = 0; i < n_layers; i++) {
			for (int j = 0; j < (int)layers_info_.layers[i]->outputs().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->outputs()[j]);
				top_ref_count[h]++;
			}
		}
		for (int i = n_layers - 1; i >= 0; i--) {
			unsigned int bp_release_top_diff_mask = 0, bp_release_top_data_mask = 0;
			for (int j = 0; j < (int)layers_info_.layers[i]->outputs().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->outputs()[j]);
				top_ref_count[h]--;
				if (top_ref_count[h] == 0) {
					if (!layers_info_.layers[i]->outputs()[j]->get_attr(node_blob::NF_NET_INPUT)) {
						bp_release_top_diff_mask |= 1U << j;
					}
					if (!layers_info_.layers[i]->outputs()[j]->get_attr(node_blob::NF_NET_OUTPUT)) {
						bp_release_top_data_mask |= 1U << j;
					}
				}
			}
			layers_info_.runtime_properties[i].bp_release_top_diff_mask = bp_release_top_diff_mask;
			layers_info_.runtime_properties[i].bp_release_top_data_mask = bp_release_top_data_mask;
		}

		// generate bp clear diff mask for weights and bottoms
		set<blob_hash> first_visited_weights, first_visited_bottoms;
		for (int i = n_layers - 1; i >= 0; i--) {
			unsigned int mask_clear_weights = 0;
			for (int j = 0; j < (int)layers_info_.layers[i]->weights().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->weights()[j]);
				if (!first_visited_weights.count(h) 
					&& should_bp(layers_info_.runtime_properties[i].mask_bp_weights, j)) {
					mask_clear_weights |= 1U << j;
					first_visited_weights.insert(h);
				}
			}
			layers_info_.runtime_properties[i].mask_clear_weights = mask_clear_weights;

			unsigned int mask_clear_acts = 0;
			for (int j = 0; j < (int)layers_info_.layers[i]->inputs().size(); j++) {
				blob_hash h = get_hash(layers_info_.layers[i]->inputs()[j]);

				if (j == 0 && layers_info_.layers[i]->inplace()) { // inplace case
					// for inplace input, we should always clear the acts
					if (should_bp(layers_info_.runtime_properties[i].mask_bp_acts, j)) {
						mask_clear_acts |= 1U << j;
					}
					else {
						if (first_visited_bottoms.count(h)) {
							first_visited_bottoms.erase(h);
						}
					}
				}
				else {
					if (!first_visited_bottoms.count(h)
						&& should_bp(layers_info_.runtime_properties[i].mask_bp_acts, j)) {
						mask_clear_acts |= 1U << j;
						first_visited_bottoms.insert(h);
					}
				}
			}
			layers_info_.runtime_properties[i].mask_clear_acts = mask_clear_acts;
		}
	}

	void caffepro_net::setup_weights(const NetParameter &param) {
		map<blob_hash, int> weight_hash_to_index;

		weights_info_.weights.clear();
		weights_info_.learning_rate.clear();
		weights_info_.weight_decay.clear();

		CHECK_EQ(layers_info_.layers.size(), param.layers_size());
		for (int i = 0; i < (int)layers_info_.layers.size(); i++) {
			for (int j = 0; j < (int)layers_info_.layers[i]->weights().size(); j++) {
				auto &weight = layers_info_.layers[i]->weights()[j];
				blob_hash h = get_hash(weight);

				data_type current_lr = (j >= param.layers(i).layer().blobs_lr_size())
					? (data_type)1.f : param.layers(i).layer().blobs_lr(j);
				data_type current_wc = (j >= param.layers(i).layer().weight_decay_size())
					? (data_type)1.f : param.layers(i).layer().weight_decay(j);

				if (weight_hash_to_index.count(h)) {
					LOG(INFO) << "Shared weights detected in " << param.layers(i).layer().name() << "[" << j << "]";
					int shared_index = weight_hash_to_index[h];

					if (current_lr != 0
						&& weights_info_.learning_rate[shared_index] != 0
						&& weights_info_.learning_rate[shared_index] != current_lr) {
						LOG(ERROR) << "Warning: shared weights learning rate mismatch. Use the larger one";
					}

					weights_info_.learning_rate[shared_index] = std::max(weights_info_.learning_rate[shared_index], current_lr);

					if (current_wc != 0
						&& weights_info_.weight_decay[shared_index] != 0
						&& weights_info_.weight_decay[shared_index] != current_wc) {
						LOG(ERROR) << "Warning: shared weights weight decay mismatch. Use the larger one";
					}

					weights_info_.weight_decay[shared_index] = std::max(weights_info_.weight_decay[shared_index], current_wc);
				}
				else {
					weights_info_.weights.push_back(weight);
					weights_info_.learning_rate.push_back(current_lr);
					weights_info_.weight_decay.push_back(current_wc);
					weight_hash_to_index[h] = (int)weights_info_.weights.size() - 1;
				}
			}
		}

		LOG(INFO) << "Initialized " << weights_info_.weights.size() << " distinct weights in total";
	}

	void caffepro_net::check_topology() {
		// check inplace usage
		set<blob_hash> visible_blobs;
		for (int i = 0; i < (int)blobs_info_.input_blobs.size(); i++) {
			visible_blobs.insert(get_hash(blobs_info_.input_blobs[i]));
		}
		for (int i = 0; i < (int)layers_info_.layers.size(); i++) {
			if (layers_info_.layers[i]->inplace()) {
				CHECK(visible_blobs.count(get_hash(layers_info_.bottoms[i][0])))
					<< "Illegal inplace usage: " << layers_info_.layers[i]->layer_param().name();
			}
			for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
				visible_blobs.erase(get_hash(layers_info_.bottoms[i][j]));
			}
			for (int j = 0; j < (int)layers_info_.tops[i].size(); j++) {
				visible_blobs.insert(get_hash(layers_info_.tops[i][j]));
			}
		}
	}

	void caffepro_net::forward(bool save_memory, bool fwd_dataprovider) {
		if (fwd_dataprovider) {
			forward_data_provider();
		}

		for (int i = 0; i < (int)layers_info_.layers.size(); i++) {
			layers_info_.layers[i]->resize();
			layers_info_.layers[i]->forward();

			if (save_memory) {
				unsigned int release_mask = layers_info_.runtime_properties[i].fp_release_bottom_mask;
				for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
					if (should_release(release_mask, j)) {
						layers_info_.bottoms[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}
		}
		finished_reshape();
	}

	void caffepro_net::forward_debug(bool save_memory, bool fwd_dataprovider) {
		// init debug storage
		if (!debug_outputs_fwd_) {
			debug_outputs_fwd_.reset(new node_blob());
			debug_outputs_fwd_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(context_, 1, 1, 1, 1)));
			blobs_info_.output_blobs.push_back(debug_outputs_fwd_);
		}

		if (fwd_dataprovider) {
			forward_data_provider();
		}

		vector<std::pair<string, data_type> > metrics;

		for (int i = 0; i < (int)layers_info_.layers.size(); i++) {
			layers_info_.layers[i]->resize();
			layers_info_.layers[i]->forward();

			const string &layer_name = layers_info_.layers[i]->layer_param().name();
			for (int j = 0; j < (int)layers_info_.layers[i]->weights().size(); j++) {
				string title = layer_name + ".weights[" + std::to_string(j) + "] ";
				metrics.push_back(make_pair(
					title + "mean",
					(data_type)layers_info_.layers[i]->weights()[j]->get(0)->mean()
					));
				metrics.push_back(make_pair(
					title + "var",
					(data_type)layers_info_.layers[i]->weights()[j]->get(0)->variance()
					));
			}

			for (int j = 0; j < (int)layers_info_.layers[i]->outputs().size(); j++) {
				string title = layer_name + ".outputs[" + std::to_string(j) + "] ";
				metrics.push_back(make_pair(
					title + "mean",
					(data_type)layers_info_.layers[i]->outputs()[j]->get(0)->mean()
					));
				metrics.push_back(make_pair(
					title + "var",
					(data_type)layers_info_.layers[i]->outputs()[j]->get(0)->variance()
					));
			}

			if (save_memory) {
				unsigned int release_mask = layers_info_.runtime_properties[i].fp_release_bottom_mask;
				for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
					if (should_release(release_mask, j)) {
						layers_info_.bottoms[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}
		}

		finished_reshape();

		// write debug info back
		debug_outputs_fwd_->get(0)->reshape_4d((int)metrics.size(), 1, 1, 1);
		debug_outputs_fwd_->tags().clear();
		data_type *debug_data = debug_outputs_fwd_->get(0)->mutable_cpu_data();
		for (int i = 0; i < (int)metrics.size(); i++) {
			debug_outputs_fwd_->tags().push_back(metrics[i].first);
			debug_data[i] = metrics[i].second;
		}
	}

	void caffepro_net::forward_until(const std::string &last_layer_name, bool save_memory) {
		CHECK(has_layer(last_layer_name));
		int last_layer_index = layers_info_.layer_name_to_idx[last_layer_name];
		for (int i = 0; i <= last_layer_index; i++) {
			layers_info_.layers[i]->resize();
			layers_info_.layers[i]->forward();

			if (save_memory && i != last_layer_index) {
				unsigned int release_mask = layers_info_.runtime_properties[i].fp_release_bottom_mask;
				for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
					if (should_release(release_mask, j)) {
						layers_info_.bottoms[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}
		}
	}


	void caffepro_net::forward_range(const std::string &start_layer, const std::string &end_layer, bool save_memory) {
		CHECK(has_layer(start_layer));
		CHECK(has_layer(end_layer));

		int start_layer_index = layers_info_.layer_name_to_idx[start_layer];
		int end_layer_index = layers_info_.layer_name_to_idx[end_layer];

		for (int i = start_layer_index; i <= end_layer_index; i++) {
			string layer_name = layers_info_.layers[i]->layer_param().name();
			layers_info_.layers[i]->resize();
			layers_info_.layers[i]->forward();

			if (save_memory && i != start_layer_index && i != end_layer_index) {
				unsigned int release_mask = layers_info_.runtime_properties[i].fp_release_bottom_mask;
				for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
					if (should_release(release_mask, j)) {
						layers_info_.bottoms[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}
		}
	}

	void caffepro_net::forward_data_provider() {
		if (data_provider_) {
			data_provider_->set_namespace(name_);
			data_provider_->forward();
			context_->sync_all_devices();
			data_provider_->begin_prefetch();
		}
	}

	void caffepro_net::backward(bool save_memory, bool never_clear_weights) {
		for (int i = (int)layers_info_.layers.size() - 1; i >= 0; i--) {
			string layer_name = layers_info_.layers[i]->layer_param().name();

			auto &prop = layers_info_.runtime_properties[i];
			if (prop.mask_bp_acts || prop.mask_bp_weights) {
				layers_info_.layers[i]->backward(prop.mask_bp_acts, prop.mask_bp_weights, prop.mask_clear_acts, never_clear_weights ? 0 : prop.mask_clear_weights);
			}

			if (save_memory) {
				for (int j = 0; j < (int)layers_info_.tops[i].size(); j++) {
					if (should_release(prop.bp_release_top_diff_mask, j)) {
						layers_info_.tops[i][j]->release_diff();
					}
					if (should_release(prop.bp_release_top_data_mask, j)) {
						layers_info_.tops[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}
		}
	}

	void caffepro_net::backward_debug(bool save_memory, bool never_clear_weights) {
		// init debug storage
		if (!debug_outputs_bwd_) {
			debug_outputs_bwd_.reset(new node_blob());
			debug_outputs_bwd_->add(boost::shared_ptr<device_blob>(device_blob::create_4d(context_, 1, 1, 1, 1)));
			blobs_info_.output_blobs.push_back(debug_outputs_bwd_);
		}

		vector<std::pair<string, data_type> > metrics;

		for (int i = (int)layers_info_.layers.size() - 1; i >= 0; i--) {
			auto &prop = layers_info_.runtime_properties[i];
			if (prop.mask_bp_acts || prop.mask_bp_weights) {
				layers_info_.layers[i]->backward(prop.mask_bp_acts, prop.mask_bp_weights, prop.mask_clear_acts, never_clear_weights ? 0 : prop.mask_clear_weights);
			
				const string &layer_name = layers_info_.layers[i]->layer_param().name();
				for (int j = 0; j < (int)layers_info_.layers[i]->weights().size(); j++) {
					string title = layer_name + ".weights_diff[" + std::to_string(j) + "] ";
					metrics.push_back(make_pair(
						title + "mean",
						(data_type)layers_info_.layers[i]->weights()[j]->get(0)->mean(true)
						));
					metrics.push_back(make_pair(
						title + "var",
						(data_type)layers_info_.layers[i]->weights()[j]->get(0)->variance(true)
						));
				}

				for (int j = 0; j < (int)layers_info_.layers[i]->inputs().size(); j++) {
					string title = layer_name + ".inputs_diff[" + std::to_string(j) + "] ";
					metrics.push_back(make_pair(
						title + "mean",
						(data_type)layers_info_.layers[i]->inputs()[j]->get(0)->mean(true)
						));
					metrics.push_back(make_pair(
						title + "var",
						(data_type)layers_info_.layers[i]->inputs()[j]->get(0)->variance(true)
						));
				}
			}

			if (save_memory) {
				for (int j = 0; j < (int)layers_info_.tops[i].size(); j++) {
					if (should_release(prop.bp_release_top_diff_mask, j)) {
						layers_info_.tops[i][j]->release_diff();
					}
					if (should_release(prop.bp_release_top_data_mask, j)) {
						layers_info_.tops[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}
		}

		// write debug info back
		debug_outputs_bwd_->get(0)->reshape_4d((int)metrics.size(), 1, 1, 1);
		debug_outputs_bwd_->tags().clear();
		data_type *debug_data = debug_outputs_bwd_->get(0)->mutable_cpu_data();
		for (int i = 0; i < (int)metrics.size(); i++) {
			debug_outputs_bwd_->tags().push_back(metrics[i].first);
			debug_data[i] = metrics[i].second;
		}
	}

	void caffepro_net::forward_benchmark(bool save_memory) {
		for (int i = 0; i < (int)layers_info_.layers.size(); i++) {
			context_->sync_all_devices();
			clock_t start_time = clock();

			layers_info_.layers[i]->resize();
			layers_info_.layers[i]->forward();

			if (save_memory) {
				unsigned int release_mask = layers_info_.runtime_properties[i].fp_release_bottom_mask;
				for (int j = 0; j < (int)layers_info_.bottoms[i].size(); j++) {
					if (should_release(release_mask, j)) {
						layers_info_.bottoms[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}

			context_->sync_all_devices();
			LOG(INFO) << "(" << layers_info_.layers[i]->layer_param().name() << ")"
					  << "FP time: " << clock() - start_time;
		}
	}

	void caffepro_net::backward_benchmark(bool save_memory) {
		for (int i = (int)layers_info_.layers.size() - 1; i >= 0; i--) {
			context_->sync_all_devices();
			clock_t start_time = clock();

			auto &prop = layers_info_.runtime_properties[i];
			if (prop.mask_bp_acts || prop.mask_bp_weights) {
				layers_info_.layers[i]->backward(prop.mask_bp_acts, prop.mask_bp_weights, prop.mask_clear_acts, prop.mask_clear_weights);
			}

			if (save_memory) {
				for (int j = 0; j < (int)layers_info_.tops[i].size(); j++) {
					if (should_release(prop.bp_release_top_diff_mask, j)) {
						layers_info_.tops[i][j]->release_diff();
					}
					if (should_release(prop.bp_release_top_data_mask, j)) {
						layers_info_.tops[i][j]->release_data();
					}
				}
			}

			// release internal workspace
			auto &internal_blobs = layers_info_.layers[i]->internal_weights();
			for (int j = 0; j < (int)internal_blobs.size(); j++) {
				if (internal_blobs[j]->get_attr(node_blob::NF_TEMP)) {
					internal_blobs[j]->release_data();
					internal_blobs[j]->release_diff();
				}
			}

			context_->sync_all_devices();
			LOG(INFO) << "(" << layers_info_.layers[i]->layer_param().name() << ")"
				<< "BP time: " << clock() - start_time;
		}
	}

	void caffepro_net::finished_reshape() {
		for (int i = 0; i < (int)blobs_info_.blobs.size(); i++) {
			blobs_info_.blobs[i]->finish_reshape();
		}
	}

	void caffepro_net::load_weights_from_info(NetParameter &param, const analyzer_proto::Info &info) {

		set<blob_hash> loaded_weight_blobs;


		//for (int j = 0; j < (int)info.layers_size(); j++) {
		//	if (info.layers(j).type() != "batch_norm") {
		//		std::cout << j + 1 << "/" << info.layers_size() << " : " << info.layers(j).name() << std::endl;
		//	}
		//	// LOG(INFO) << j+1 << "/" << info.layers_size() << " : " << info.layers(j).name() << std::endl;
		//}

		for (int i = 0; i < (int)param.layers_size(); i++) {
			LayerParameter& source_layer = *param.mutable_layers(i)->mutable_layer();
			const string& source_layer_name = source_layer.name();
			if (!layers_info_.layer_name_to_idx.count(source_layer_name)) {
				LOG(INFO) << "Ignoring source layer " << source_layer_name;
			}
			else {
				int matchedLid = -1;
				int target_layer_id = layers_info_.layer_name_to_idx[source_layer_name];
				
				for (int j = 0; j < (int)info.layers_size(); j++) {
					if ((info.layers(j).name() == source_layer.name()) && (info.layers(j).type() == source_layer.type())) {
						matchedLid = j;
						break;
					}
				}

				caffepro_layer::layer_weights& target_weights = layers_info_.layers[target_layer_id]->weights();

				CHECK_EQ(target_weights.size(), source_layer.blobs_size())
					<< "Incompatible number of blobs for layer " << source_layer_name;


				for (int j = 0; j < (int)target_weights.size(); j++) {
					blob_hash h = get_hash(target_weights[j]);
					if (!loaded_weight_blobs.count(h)) {
						loaded_weight_blobs.insert(h);
						auto blob = source_layer.mutable_blobs(j);
						auto count = blob->num() * blob->channels() * blob->width() * blob->height();
						auto mlayer = info.layers(matchedLid+j);
						/*LOG(INFO) << i + 1 << "/" << param.layers_size() << " " << target_weights.size() << ": " << source_layer.name() << " "
						<< source_layer.type() << " ; " << mlayer->name() << " " << mlayer->type() << " " <<
						mlayer->weight_size() << " " << count << std::endl;*/
						CHECK_EQ(mlayer.weight_size(), count);
						blob->mutable_data()->CopyFrom(mlayer.weight());
						target_weights[j]->load_data_from(source_layer.mutable_blobs(j));
					}
					else {
						LOG(INFO) << "Skipped loading " << source_layer_name << "[" << j
							<< "] because another shared weights have been loaded";
					}
				}

			}
		}
		int kk = 10;
		kk++;
	}

	void caffepro_net::load_weights(NetParameter &param) {
		set<blob_hash> loaded_weight_blobs;
		set<string> loaded_layer_names;

		for (int i = 0; i < (int)param.layers_size(); i++) {
			LayerParameter& source_layer = *param.mutable_layers(i)->mutable_layer();
			const string& source_layer_name = source_layer.name();

			if (!layers_info_.layer_name_to_idx.count(source_layer_name)) {
				LOG(INFO) << "Ignoring source layer " << source_layer_name;
			}
			else {
				int target_layer_id = layers_info_.layer_name_to_idx[source_layer_name];
				caffepro_layer::layer_weights& target_weights = layers_info_.layers[target_layer_id]->weights();

				CHECK_EQ(target_weights.size(), source_layer.blobs_size()) 
					<< "Incompatible number of blobs for layer " << source_layer_name;

				for (int j = 0; j < (int)target_weights.size(); j++) {
					blob_hash h = get_hash(target_weights[j]);
					if (!loaded_weight_blobs.count(h)) {
						loaded_weight_blobs.insert(h);
						target_weights[j]->load_data_from(source_layer.mutable_blobs(j));
					}
					else {
						LOG(INFO) << "Skipped loading " << source_layer_name << "[" << j 
							<< "] because another shared weights have been loaded";
					}
				}

				loaded_layer_names.insert(source_layer_name);
			}
		}

		for (int i = 0; i < (int)layers().size(); i++) {
			caffepro_layer &current_layer = *layers()[i];

			if (current_layer.layer_param().blob_source_layer_size() > 0) {
				CHECK_EQ(current_layer.layer_param().blob_source_layer_size(), 1)
					<< "Currently only one blob source layer is allowed";

				

				if (!loaded_layer_names.count(current_layer.layer_param().name())) {
					const string &src_layer_name = current_layer.layer_param().blob_source_layer(0);
					CHECK(has_layer(src_layer_name)) << "Unknown source layer name: " << src_layer_name;

					caffepro_layer &src_layer = *get_layer(src_layer_name);

					caffepro_layer::layer_weights& target_weights = current_layer.weights();
					caffepro_layer::layer_weights& src_weights = src_layer.weights();

					CHECK_EQ(target_weights.size(), src_weights.size())
						<< "Incompatible number of blobs for source layer " << src_layer_name;

					for (int j = 0; j < (int)target_weights.size(); j++) {
						CHECK_EQ(target_weights[j]->get(0)->count(), src_weights[j]->get(0)->count());
						context_->sync_all_devices();
						target_weights[j]->get(0)->copy_data_from_via_gpu(*src_weights[j]->get(0));
						target_weights[j]->broadcast_data_via_gpu(0);
					}

					LOG(INFO) << "Loaded " << current_layer.layer_param().name() << " from " << src_layer_name << " layer";
				}
				else {
					LOG(INFO) << "Skipped loading " << current_layer.layer_param().name()
						<< " from source layer(s) because it has been loaded from model file";
				}
			}
		}
	}

	void caffepro_net::save_proto(NetParameter &param) {
		param.Clear();
		param.set_name(name_);

		for (int i = 0; i < (int)blobs_info_.input_blobs.size(); ++i) {
			param.add_input(blobs_info_.input_blobs[i]->get_name());
		}
		DLOG(INFO) << "Serializing " << layers_info_.layers.size() << " layers";
		for (int i = 0; i < (int)layers_info_.layers.size(); ++i) {
			LayerConnection* layer_connection = param.add_layers();
			for (int j = 0; j < (int)layers_info_.bottoms[i].size(); ++j) {
				layer_connection->add_bottom(layers_info_.bottoms[i][j]->get_name());
			}
			for (int j = 0; j < (int)layers_info_.tops[i].size(); ++j) {
				layer_connection->add_top(layers_info_.tops[i][j]->get_name());
			}
			LayerParameter* layer_parameter = layer_connection->mutable_layer();
			layers_info_.layers[i]->write_to_proto(layer_parameter);
		}
	}

	void caffepro_net::release_blobs() {
		for (int i = 0; i < (int)blobs_info_.blobs.size(); i++) {
			if (!blobs_info_.blobs[i]->get_attr(node_blob::NF_NET_INPUT)
				&& !blobs_info_.blobs[i]->get_attr(node_blob::NF_NET_OUTPUT))
			blobs_info_.blobs[i]->release_data();
			blobs_info_.blobs[i]->release_diff();
		}
	}

	void caffepro_net::share_weights_from(caffepro_net &other) {
		for (auto iter = other.layers().begin(); iter != other.layers().end(); ++iter) {
			if (has_layer((*iter)->layer_param().name())) {
				caffepro_layer &layer = *get_layer((*iter)->layer_param().name());
				CHECK_EQ(layer.weights().size(), (*iter)->weights().size());
				CHECK_EQ(layer.internal_weights().size(), (*iter)->internal_weights().size());
				for (int i = 0; i < (int)layer.weights().size(); i++) {
					node_blob &this_weight = *layer.weights()[i];
					node_blob &other_weight = *(*iter)->weights()[i];
					CHECK_EQ(this_weight.size(), other_weight.size());
					for (int nd = 0; nd < (int)this_weight.size(); nd++) {
						CHECK_EQ(this_weight[nd]->device_id(), other_weight[nd]->device_id());
						CHECK_EQ(this_weight[nd]->count(), other_weight[nd]->count());
						this_weight[nd]->share_data(*other_weight[nd]);
					}
				}
				for (int i = 0; i < (int)layer.internal_weights().size(); i++) {
					node_blob &this_weight = *layer.internal_weights()[i];
					node_blob &other_weight = *(*iter)->internal_weights()[i];
					if (!this_weight.get_attr(node_blob::NF_TEMP)) {
						CHECK_EQ(this_weight.size(), other_weight.size());
						for (int nd = 0; nd < (int)this_weight.size(); nd++) {
							CHECK_EQ(this_weight[nd]->device_id(), other_weight[nd]->device_id());
							CHECK_EQ(this_weight[nd]->count(), other_weight[nd]->count());
							this_weight[nd]->share_data(*other_weight[nd]);
						}
					}
				}
			}
		}
	}

	void caffepro_net::mark_as_input_blob(const std::string &blob_name, bool reset_layer_runtime_properties) {
		CHECK(has_blob(blob_name));
		get_blob(blob_name)->set_attr(node_blob::NF_NET_INPUT);
		if (reset_layer_runtime_properties) {
			setup_layer_runtime_properties();
		}
	}

	void caffepro_net::mark_as_output_blob(const std::string &blob_name, bool reset_layer_runtime_properties) {
		CHECK(has_blob(blob_name));
		get_blob(blob_name)->set_attr(node_blob::NF_NET_OUTPUT);
		if (reset_layer_runtime_properties) {
			setup_layer_runtime_properties();
		}
	}
}