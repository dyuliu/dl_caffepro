
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/utils/random_helper.h>
#include <caffepro/utils/data_utils/box.h>

using std::vector;
using std::string;

namespace caffepro {

	using data_utils::crop_type;

	void* data_bigfile_layer::data_prefetch(void* layer_pointer) {

		clock_t clk_start = clock();
		// set rand seed, for each is in a new thread
		srand(std::random_device()());

		CHECK(layer_pointer);
		data_bigfile_layer* layer = static_cast<data_bigfile_layer *>(layer_pointer);		
		CHECK(layer);

		// parameters
		auto &layer_param = layer->layer_param_.data_bigfile_param();
		const data_type scale = layer_param.scale();
		const int batchsize = layer_param.batch_size();
		int channel_num_total = layer_param.channel_num(0);
		crop_type croptype = crop_type(layer_param.crop_type());
		const bool cache_data = layer_param.cache_data();

		// data sources
		bigfile_database &database = layer->database_;
		vector<data_utils::raw_picture> &imgs = database.imgs;
		cv::Mat &data_mean = database.data_mean;

		// prefetch data pool
		batch_prefetch &prefetch = layer->prefetch_;

		// image pool
		vector<cv::Mat> image_data(batchsize);

		// label pool
		data_type* label_data = prefetch.labels->get(0)->mutable_cpu_data();
		memset(label_data, 0, prefetch.labels->get(0)->count() * sizeof(data_type)); // clear label

		// prefetch state
		batch_state &prefetch_state = prefetch.prefetch_batch_state;
		prefetch_state.processing_imgs.resize(batchsize);
		int start_img_idx = prefetch_state.image_start_idx;

		// move buffer heads to cpu
		for (int nd = 0; nd < (int)prefetch.images->size(); nd++) {
			prefetch.images->get(nd)->write_only_cpu_data();
		}
		for (int nd = 0; nd < (int)prefetch.labels->size(); nd++) {
			prefetch.labels->get(nd)->write_only_cpu_data();
		}
		for (int i = 0; i < (int)prefetch.extra_data.size(); i++) {
			for (int nd = 0; nd < (int)prefetch.extra_data[i]->size(); nd++) {
				prefetch.extra_data[i]->get(nd)->write_only_cpu_data();
			}
		}

#pragma omp parallel for schedule(dynamic)
		for (int itemid = 0; itemid < batchsize; itemid++) {
			int img_idx = (start_img_idx + itemid) % (int)imgs.size();

			if (!cache_data) {
				imgs[img_idx].load_data();
			}

			layer->prepare_one_image(image_data[itemid], imgs[img_idx], itemid, channel_num_total, data_mean, prefetch_state.fixed_view);
			CHECK_EQ(image_data[itemid].channels(), channel_num_total) << "Prepared image.channels() != sum(channel_num)";

			// write back label
			if (database.multilabel_classes <= 0) { // single label
				label_data[itemid] = static_cast<data_type>(imgs[img_idx].label_id);
			}
			else { // multi label
				vector<int> labels;
				if (database.picname_to_multilabel.count(imgs[img_idx].path)) {
					labels = database.picname_to_multilabel[imgs[img_idx].path];
				}
				else if (database.clsname_to_multilabel.count(database.class_info.classID2label[imgs[img_idx].label_id])) {
					labels = database.clsname_to_multilabel[database.class_info.classID2label[imgs[img_idx].label_id]];
				}
				else {
					LOG(FATAL) << "No multi-label info for the picture: " << imgs[img_idx].path;
				}

				CHECK_GT(labels.size(), 0);

				data_type *cur_top_label = label_data + prefetch.labels->get(0)->offset(itemid);
				for (vector<int>::iterator iter = labels.begin(); iter != labels.end(); ++iter) {
					cur_top_label[*iter] = (data_type)1.f;
				}
			}

			if (!cache_data) { // release loaded image to same memory
				imgs[img_idx].clear_data();
			}
		}

		// write back images
		layer->write_image_to_blob(croptype, image_data, prefetch.images);

		string strPhase = (layer->context_->get_phase() == caffepro_context::TRAIN) ? "Train" : "Test";
		// move to next data
		if (layer->layer_param_.data_bigfile_param().random_shuffle() 
			&& prefetch_state.image_start_idx + batchsize >= (int)imgs.size()) {
			std::random_shuffle(imgs.begin(), imgs.end());
			// We have reached the end. Restart from the first.
			DLOG(INFO) << strPhase << ": restarting data prefetching from start.";
		}

		if (prefetch_state.image_start_idx == prefetch_state.image_start_idx + batchsize) {
			LOG(INFO) << strPhase << ": time for Data Prefetch " << (float)(clock() - clk_start) / 1000 << "s";
		}

		prefetch_state.image_start_idx = (prefetch_state.image_start_idx + batchsize) % (int)imgs.size();
		return nullptr;
	}

	void data_bigfile_layer::write_image_to_blob(data_utils::crop_type croptype, const std::vector<cv::Mat> &images, boost::shared_ptr<node_blob> blob) {
		CHECK_EQ(images.size(), blob->sum_num());

		bool same_shape = true;
		if (!is_fixed_size(croptype)) {
			for (int nd = 0, start = 0; nd < (int)blob->size(); nd++) {
				for (int i = start + 1; i < start + blob->get(nd)->num(); i++) {
					if (images[i].rows != images[start].rows
						|| images[i].cols != images[start].cols
						|| images[i].channels() != images[start].channels()) {
						same_shape = false;
						break;
					}
				}
				if (!same_shape) break;
				start += blob->get(nd)->num();
			}

			if (same_shape) { // same shape per device
				for (int nd = 0, start = 0; nd < (int)blob->size(); nd++) {
					blob->get(nd)->reshape_4d(blob->get(nd)->num(), images[start].channels(), images[start].rows, images[start].cols);
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
					const cv::Mat &im = images[img_index];
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
#pragma omp parallel for
			for (int img_index = 0; img_index < (int)images.size(); img_index++) {
				int sum = 0, nd = 0;
				while (true) {
					CHECK_LT(nd, blob->size());
					if (sum + blob->get(nd)->num() > img_index) break;
					sum += blob->get(nd)->num();
					nd++;
				}
				int n = img_index - sum;

				const cv::Mat &im = images[img_index];
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