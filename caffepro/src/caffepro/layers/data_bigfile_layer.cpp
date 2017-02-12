
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/context/common_names.h>

using std::string;
using std::vector;

namespace caffepro {
	data_bigfile_layer::data_bigfile_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 0;
		attr_.num_outputs_min = 2;
		attr_.num_outputs_max = INT_MAX;
		attr_.device_dispatcher_forward = layer_attribute::OUTPUT_BASE;
		attr_.device_dispatcher_backward = layer_attribute::OUTPUT_BASE;
		attr_.usage = layer_attribute::USAGE_DATA_SOURCE;

		// no constraint needed here
	}

	data_bigfile_layer::~data_bigfile_layer() {
		release_all();
	}

	void data_bigfile_layer::init() {
		data_prefetch_event_name_ = "data_bigfile_prefetch";

		int batchsize = layer_param_.data_bigfile_param().batch_size();
		int batchimgsize = layer_param_.data_bigfile_param().batch_img_size();
		string source = layer_param_.data_bigfile_param().source();
		CHECK_EQ(layer_param_.data_bigfile_param().channel_num_size(), 1);
		int channel_num = layer_param_.data_bigfile_param().channel_num(0);
		
		database_.imgs.clear();
		database_.data_mean = cv::Mat();
		database_.class_info = data_utils::kl_infos();
		database_.multilabel_classes = -1;
		database_.clsname_to_multilabel.clear();
		database_.picname_to_multilabel.clear();

		// init database
		init_picture_database();

		// output blobs: [picture (1), label (1), extra outputs (determined by additional_data_processer)]
		int data_extra_num = 0;
		for (int i = 0; i < this->layer_param_.data_bigfile_param().additional_data_processer_size(); i++) {
			auto &processer = this->layer_param_.data_bigfile_param().additional_data_processer(i);
			for (int j = 0; j < processer.binding_output_index_size(); j++) {
				int binding_idx = processer.binding_output_index(j);
				data_extra_num = std::max(data_extra_num, binding_idx); // find largest index
			}
		}
		if (data_extra_num) {
			data_extra_num = (data_extra_num + 1) /* number of outputs */ - 1 /* data outputs */ - 1 /* label output */;
		}

		CHECK_EQ(1 + 1 + data_extra_num, outputs_.size()) << "Data Layer takes (1 + 1 + additional_data) blobs as output.";

		// init gpu split attribute
		int n_gpus = 1;

		num_start_.clear();
		num_size_.clear();
		std::vector<int> split_device_ids;
		if (layer_param_.has_gpu_split()) {
			CHECK_EQ(layer_param_.gpu_split().split_minibatch_size(), layer_param_.gpu_split().split_gpu_id_size());
			n_gpus = layer_param_.gpu_split().split_minibatch_size();

			int sum_num = 0;
			for (int nd = 0; nd < n_gpus; nd++) {
				num_start_.push_back(sum_num);
				num_size_.push_back(layer_param_.gpu_split().split_minibatch(nd));
				split_device_ids.push_back(layer_param_.gpu_split().split_gpu_id(nd));
				sum_num += num_size_[nd];
			}

			CHECK_EQ(sum_num, batchsize);
		}
		else {
			num_start_.push_back(0);
			num_size_.push_back(batchsize);
			split_device_ids.push_back(context_->get_current_device()->device_id());
		}

		// init data output
		prefetch_.images.reset(new node_blob());
		for (int nd = 0; nd < n_gpus; nd++) {
			prefetch_.images->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
					nullptr, num_size_[nd], channel_num, batchimgsize, batchimgsize, split_device_ids[nd]
					) // do not use managered device blob because memory manager is not thread safe
				));
		}

		// init label output
		prefetch_.labels.reset(new node_blob());
		if (database_.multilabel_classes <= 0) { // single label

			prefetch_.labels->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
						nullptr, batchsize, 1, 1, 1, context_->get_current_device()->device_id()
					) // do not use managered device blob because memory manager is not thread safe
				));
		}
		else { // multi label
			prefetch_.labels->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(
					nullptr, batchsize, database_.multilabel_classes, 1, 1, context_->get_current_device()->device_id()
					) // do not use managered device blob because memory manager is not thread safe
				));
		}

		// init extra data outputs
		setup_extra_data_processer(data_extra_num);

		// set output layer attributes
		for (int i = 0; i < (int)outputs_.size(); i++) {
			outputs_[i]->set_attr(node_blob::NF_NET_INPUT);
		}

		// output debug info
		string strPhase = (context_->get_phase() == caffepro_context::TRAIN) ? "Train" : "Test";
		LOG(INFO) << strPhase << ": source data: " << this->layer_param_.data_bigfile_param().source();
		LOG(INFO) << strPhase << ": Total Image: " << database_.imgs.size() << ", Per Iter Num: " << batchsize << " -> " << float(database_.imgs.size()) / batchsize << " Iters per Epoch";

		// run prefetch thread
		DLOG(INFO) << "Initializing prefetch";
		data_prefetch(this);

		DLOG(INFO) << "Prefetch initialized.";
	}

	void data_bigfile_layer::resize() {
		context_->events()->wait(data_prefetch_event_name_);

		if (outputs_[0]->size() == 0) {
			outputs_[0]->add_like(context_, *prefetch_.images);
			outputs_[1]->add_like(context_, *prefetch_.labels);
			for (int i = 2; i < (int)outputs_.size(); i++) {
				outputs_[i]->add_like(context_, *prefetch_.extra_data[i - 2]);
			}
		}
		else {
			for (int nd = 0; nd < (int)prefetch_.images->size(); nd++) {
				if (prefetch_.images->get(nd)->reshaped()) {
					outputs_[0]->get(nd)->reshape_like(*prefetch_.images->get(nd));
					prefetch_.images->get(nd)->finish_reshape();
				}
			}
			for (int nd = 0; nd < (int)prefetch_.labels->size(); nd++) {
				if (prefetch_.labels->get(nd)->reshaped()) {
					outputs_[1]->get(nd)->reshape_like(*prefetch_.labels->get(nd));
					prefetch_.labels->get(nd)->finish_reshape();
				}
			}
			for (int i = 2; i < (int)outputs_.size(); i++) {
				for (int nd = 0; nd < (int)prefetch_.extra_data[i - 2]->size(); nd++) {
					if (prefetch_.extra_data[i - 2]->get(nd)->reshaped()) {
						outputs_[i]->get(nd)->reshape_like(*prefetch_.extra_data[i - 2]->get(nd));
						prefetch_.extra_data[i - 2]->get(nd)->finish_reshape();
					}
				}
			}
		}

		caffepro_object *last_datasource = context_->set_shared_object(namespace_, SHAREDOBJNAME_DATASOURCE, this);
		if (last_datasource != nullptr && last_datasource != this) {
			LOG(ERROR) << "Warning: more than one data sources in the same namespace";
		}
	}

	void data_bigfile_layer::on_forward(int device_index) {
		CUDA_CHECK(cudaMemcpyAsync(
			outputs_[0]->get(device_index)->mutable_gpu_data(),
			prefetch_.images->get(device_index)->gpu_data_async(),
			prefetch_.images->get(device_index)->count() * sizeof(data_type),
			cudaMemcpyDeviceToDevice
			));

		if (device_index < (int)prefetch_.labels->size()) {
			CUDA_CHECK(cudaMemcpyAsync(
				outputs_[1]->get(device_index)->mutable_gpu_data(),
				prefetch_.labels->get(device_index)->gpu_data_async(),
				prefetch_.labels->get(device_index)->count() * sizeof(data_type),
				cudaMemcpyDeviceToDevice
				));
		}

		for (int i = 2; i < outputs_.size(); i++) {
			auto &extra_data = prefetch_.extra_data[i - 2];
			if (device_index < (int)extra_data->size()) {
				CUDA_CHECK(cudaMemcpyAsync(
					outputs_[i]->get(device_index)->mutable_gpu_data(),
					extra_data->get(device_index)->gpu_data_async(),
					extra_data->get(device_index)->count() * sizeof(data_type),
					cudaMemcpyDeviceToDevice
					));
			}
		}
	}

	void data_bigfile_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		// nothing to do
	}

	void data_bigfile_layer::on_after_forward() {
		current_batch_state_ = prefetch_.prefetch_batch_state;
		context_->sync_all_devices();

		context_->events()->create(data_prefetch_event_name_, 
			event_manager::EVENT_TYPE_PREPARE_BATCH, data_prefetch, this);
	}
}