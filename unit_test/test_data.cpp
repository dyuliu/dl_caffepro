
#include <caffepro/object_model/data_model/data_provider.h>
#include <caffepro/data/data_entries/data_entries.h>
#include <caffepro/proto/caffe.pb.h>

using namespace caffepro;
using namespace caffepro::data_model;
using namespace caffepro::data_utils;

void test_rcnn_entry() {
	caffepro_context context;
	caffepro_config config;
	config.load_config("provider_config.txt");
	data_provider provider(&context, &config);
	provider.set_data_provider_name("provider_cfg");
	provider.auto_build();

	rcnn_entry &rcnn = dynamic_cast<rcnn_entry &>(*provider.get_entry("rcnn1"));
	rcnn_entry &rcnn2 = dynamic_cast<rcnn_entry &>(*provider.get_entry("rcnn2"));
	rcnn.set_enabled(true);
	rcnn2.set_enabled(true);

	provider.init();
	provider.forward();
	provider.prefetch_batch();
	provider.forward();

	auto &batch_data = provider.current_batch()->batch_data;

	const data_type *output_data = rcnn.outputs()[0]->get(0)->cpu_data();
	rcnn.outputs()[0]->get(0)->sync_ext_dim();
	const int *output_dims = rcnn.outputs()[0]->get(0)->ext_dims_cpu();

	const data_type *output_data2 = rcnn2.outputs()[0]->get(0)->cpu_data();
	rcnn2.outputs()[0]->get(0)->sync_ext_dim();
	const int *output_dims2 = rcnn2.outputs()[0]->get(0)->ext_dims_cpu();

	for (int i = 0; i < (int)batch_data.size(); i++) {
		cv::Mat &im = *batch_data[i].processed_data->data;
		auto &original_data = batch_data[i].original_data;
		
		original_data->load_data();
		cv::Mat original_im = cv::imdecode(*original_data->data, CV_LOAD_IMAGE_COLOR);
		cv::imshow("org", original_im);

		int w = output_dims[i * 3], h = output_dims[i * 3 + 1];
		cv::Mat output_im(h, w, im.type());
		const data_type *src_data = output_data + rcnn.outputs()[0]->get(0)->offset(i);

		for (int r = 0; r < h; r++) {
			float *pPixel = reinterpret_cast<float *>(output_im.ptr(r));
			for (int c = 0; c < w; c++) {
				for (int cl = 0; cl < 3; cl++) {
					*pPixel = src_data[cl * w * h + r * w + c];
					pPixel++;
				}
			}
		}
		
		output_im.convertTo(output_im, original_im.type());
		cv::imshow("output", output_im);

		int w2 = output_dims2[i * 3], h2 = output_dims2[i * 3 + 1];
		cv::Mat output_im2(h2, w2, im.type());
		const data_type *src_data2 = output_data2 + rcnn2.outputs()[0]->get(0)->offset(i);

		for (int r = 0; r < h2; r++) {
			float *pPixel = reinterpret_cast<float *>(output_im2.ptr(r));
			for (int c = 0; c < w2; c++) {
				for (int cl = 0; cl < 3; cl++) {
					*pPixel = src_data2[cl * w2 * h2 + r * w2 + c];
					pPixel++;
				}
			}
		}

		output_im2.convertTo(output_im2, original_im.type());
		cv::imshow("output2", output_im2);

		im.convertTo(im, original_im.type());
		cv::imshow("crop", im);

		cv::waitKey();
	}
}

void test_rfcn_entry() {
	caffepro_context context;
	caffepro_config config;
	config.load_config("provider_config_rfcn.txt");
	data_provider provider(&context, &config);
	provider.set_data_provider_name("provider_cfg");
	provider.auto_build();

	rfcn_entry &rfcn = dynamic_cast<rfcn_entry &>(*provider.get_entry("rfcn1"));
	rfcn.set_enabled(true);

	provider.init();
	provider.forward();
	provider.prefetch_batch();
	provider.forward();
	provider.prefetch_batch();
	provider.forward();

	auto &batch_data = provider.current_batch()->batch_data;
	int nd = 0;

	const data_type *output_data = rfcn.outputs()[0]->get(nd)->cpu_data();
	rfcn.outputs()[0]->get(nd)->sync_ext_dim();
	const int *output_dims = rfcn.outputs()[0]->get(nd)->ext_dims_cpu();
	
	const data_type *content_box_data = rfcn.outputs()[2]->get(nd)->cpu_data();
	const data_type *label_data = rfcn.outputs()[1]->get(nd)->cpu_data();

	for (int i = 0; i < (int)batch_data.size(); i++) {
		cv::Mat &im = *batch_data[i].processed_data->data;
		auto &original_data = batch_data[i].original_data;
		auto &processed_data = batch_data[i].processed_data;

		original_data->load_data();
		cv::Mat original_im = cv::imdecode(*original_data->data, CV_LOAD_IMAGE_COLOR);
		Box gt_box = boost::any_cast<Box>(processed_data->additional_data["gt_box"]);
		cv::rectangle(original_im, cv::Rect(gt_box.left, gt_box.top, gt_box.width(), gt_box.height()), cv::Scalar(0, 0, 255));
		Box crop_box = boost::any_cast<Box>(processed_data->additional_data["crop_box"]);
		cv::rectangle(original_im, cv::Rect(crop_box.left, crop_box.top, crop_box.width(), crop_box.height()), cv::Scalar(0, 255, 0));
		BoxF context_box = boost::any_cast<BoxF>(processed_data->additional_data["context_box"]);
		cv::rectangle(original_im, cv::Rect((int)context_box.left, (int)context_box.top, (int)context_box.width(), (int)context_box.height()), cv::Scalar(255, 0, 0));
		cv::imshow("org", original_im);

		int w = output_dims[i * 3], h = output_dims[i * 3 + 1];
		cv::Mat output_im(h, w, im.type());
		const data_type *src_data = output_data + rfcn.outputs()[0]->get(nd)->offset(i);

		for (int r = 0; r < h; r++) {
			float *pPixel = reinterpret_cast<float *>(output_im.ptr(r));
			for (int c = 0; c < w; c++) {
				for (int cl = 0; cl < 3; cl++) {
					*pPixel = abs(src_data[cl * w * h + r * w + c]);
					pPixel++;
				}
			}
		}

		output_im.convertTo(output_im, original_im.type());
		
		data_utils::BoxF content_box(content_box_data + i * 4);
		cv::Rect rc((int)content_box.left, (int)content_box.top, (int)content_box.width(), (int)content_box.height());

		cv::rectangle(output_im, rc, cv::Scalar(0, 255, 0));
		cv::imshow("output", output_im);

		LOG(ERROR) << label_data[i];

		cv::waitKey();
	}
}

void test_resize() {
	caffepro_context context;

	// init provider
	caffepro_config config;
	config.load_config("provider_config_rfcn.txt");
	data_provider provider(&context, &config);
	provider.set_data_provider_name("provider_cfg");
	provider.auto_build();

	rfcn_entry &rfcn = dynamic_cast<rfcn_entry &>(*provider.get_entry("rfcn1"));
	rfcn.set_enabled(true);

	provider.init();
	provider.forward();

	caffepro_layer::layer_io_buffer grid_inputs, grid_outputs;
	grid_inputs.resize(3);
	grid_inputs[0] = rfcn.outputs()[0];
	grid_inputs[1].reset(new node_blob());
	grid_inputs[1]->add(boost::shared_ptr<device_blob>(device_blob::create_4d(
		&context, grid_inputs[0]->get(0)->num(), 1, 1, 1, 0)
		));
	grid_inputs[1]->get(0)->fill_data(1.f);
	int num = grid_inputs[1]->get(0)->num();
	for (int i = 0; i < num; i++) grid_inputs[1]->get(0)->mutable_cpu_data()[i] = i % 2 ? 1.5f : 0.75f;
	grid_inputs[2] = rfcn.outputs()[2];
	grid_outputs.resize(3);
	grid_outputs[0].reset(new node_blob());
	grid_outputs[1].reset(new node_blob());
	grid_outputs[2].reset(new node_blob());
	LayerParameter resize_layer_param;
	resize_layer_param.set_name("resize_grid1");
	resize_layer_param.set_type("resize_grid");
	boost::shared_ptr<caffepro_layer> resize_layer(caffepro_layer::create(&context, resize_layer_param));
	resize_layer->config().set_config(&config);
	resize_layer->config().set_default_section_name(resize_layer_param.name());
	resize_layer->bind(grid_inputs, grid_outputs);

	resize_layer->init();
	resize_layer->resize();
	resize_layer->forward();

	caffepro_layer::layer_io_buffer sample_inputs, sample_outputs;
	sample_inputs.resize(3);
	sample_inputs[0] = rfcn.outputs()[0];
	sample_inputs[1] = grid_outputs[0];
	sample_inputs[2] = grid_outputs[1];
	sample_outputs.resize(1);
	sample_outputs[0].reset(new node_blob());
	LayerParameter sample_param;
	sample_param.set_name("sample1");
	sample_param.set_type("sample");
	boost::shared_ptr<caffepro_layer> sample_layer(caffepro_layer::create(&context, sample_param));
	sample_layer->bind(sample_inputs, sample_outputs);

	sample_layer->init();
	sample_layer->resize();
	sample_layer->forward();

	auto &batch_data = provider.current_batch()->batch_data;
	int nd = 0;

	const data_type *output_data = sample_outputs[0]->get(nd)->cpu_data();
	sample_outputs[0]->get(nd)->sync_ext_dim();
	const int *output_dims = sample_outputs[0]->get(nd)->ext_dims_cpu();

	const data_type *label_data = rfcn.outputs()[1]->get(nd)->cpu_data();
	const data_type *content_box_data = grid_outputs[2]->get(nd)->cpu_data();

	for (int i = 0; i < (int)batch_data.size(); i++) {
		cv::Mat &im = *batch_data[i].processed_data->data;
		auto &original_data = batch_data[i].original_data;
		auto &processed_data = batch_data[i].processed_data;

		original_data->load_data();
		cv::Mat original_im = cv::imdecode(*original_data->data, CV_LOAD_IMAGE_COLOR);
		Box gt_box = boost::any_cast<Box>(processed_data->additional_data["gt_box"]);
		cv::rectangle(original_im, cv::Rect(gt_box.left, gt_box.top, gt_box.width(), gt_box.height()), cv::Scalar(0, 0, 255));
		Box crop_box = boost::any_cast<Box>(processed_data->additional_data["crop_box"]);
		cv::rectangle(original_im, cv::Rect(crop_box.left, crop_box.top, crop_box.width(), crop_box.height()), cv::Scalar(0, 255, 0));
		BoxF context_box = boost::any_cast<BoxF>(processed_data->additional_data["context_box"]);
		cv::rectangle(original_im, cv::Rect((int)context_box.left, (int)context_box.top, (int)context_box.width(), (int)context_box.height()), cv::Scalar(255, 0, 0));
		cv::imshow("org", original_im);

		int w = output_dims[i * 3], h = output_dims[i * 3 + 1];
		cv::Mat output_im(h, w, im.type());
		const data_type *src_data = output_data + sample_outputs[0]->get(nd)->offset(i);

		for (int r = 0; r < h; r++) {
			float *pPixel = reinterpret_cast<float *>(output_im.ptr(r));
			for (int c = 0; c < w; c++) {
				for (int cl = 0; cl < 3; cl++) {
					*pPixel = abs(src_data[cl * w * h + r * w + c]);
					pPixel++;
				}
			}
		}

		output_im.convertTo(output_im, original_im.type());

		data_utils::BoxF content_box(content_box_data + i * 4);
		cv::Rect rc((int)content_box.left, (int)content_box.top, (int)content_box.width(), (int)content_box.height());

		cv::rectangle(output_im, rc, cv::Scalar(0, 255, 0));
		cv::imshow("output", output_im);

		LOG(ERROR) << label_data[i];

		cv::waitKey();
	}
}

//int main() {
//	// test_rcnn_entry();
//	// test_rfcn_entry();
//	test_resize();
//}