
#include <caffepro/layers/anchor_loc_loss_ex_layer.h>
#include <caffepro/layers/sigmoid_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/layers/data_bigfile_layer.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/data_utils/bigfile.h>
#include <caffepro/utils/data_utils/box_trans.h>
#include <caffepro/math/cublas_wrapper.h>

namespace caffepro {
	using std::vector;
	using std::string;
	using data_utils::Box;
	using data_utils::box_t;

	anchor_loc_loss_ex_layer::anchor_loc_loss_ex_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 3;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_UNIQUE_DEVICE
			| layer_attribute::CF_REQUIRE_SAME_NUM
			| layer_attribute::CF_REQUIRE_SAME_DEVICE
			| layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);

		attr_.usage = layer_attribute::USAGE_LOSS;
	}

	anchor_loc_loss_ex_layer::~anchor_loc_loss_ex_layer() {
		release_all();
	}

	void anchor_loc_loss_ex_layer::init() {
		check_input();

		auto_spatial_anchor_ = layer_param_.clsloc_loss_param().auto_spatial_anchor();

		// init anchor params
		if (layer_param_.has_anchor_param()) {
			LOG(INFO) << "Using anchor params within layer defination";

			if (auto_spatial_anchor_) {
				CHECK_GT(inputs_[0]->get(0)->width(), 1);
				CHECK_GT(inputs_[0]->get(0)->height(), 1);

				anchor_num_width_ = inputs_[0]->get(0)->width();
				anchor_num_height_ = inputs_[0]->get(0)->height();
			}
			else {
				anchor_num_width_ = layer_param_.anchor_param().spatial_width();
				anchor_num_height_ = layer_param_.anchor_param().spatial_height();
			}

			CHECK(layer_param_.anchor_param().has_spatial_start());
			CHECK(layer_param_.anchor_param().has_spatial_step());

			spatial_start_ = layer_param_.anchor_param().spatial_start();
			spatial_step_ = layer_param_.anchor_param().spatial_step();

			central_scales_.clear();
			if (layer_param_.anchor_param().central_scale_size() > 0) {
				for (int i = 0; i < layer_param_.anchor_param().central_scale_size(); i++) {
					central_scales_.push_back(layer_param_.anchor_param().central_scale(i));
				}
			}
			else {
				central_scales_.push_back((data_type)1.f); // default scale anchor
			}

			aspect_ratio_.clear();
			if (layer_param_.anchor_param().aspect_ratio_size() > 0) {
				for (int i = 0; i < layer_param_.anchor_param().aspect_ratio_size(); i++) {
					aspect_ratio_.push_back(layer_param_.anchor_param().aspect_ratio(i));
				}
			}
			else {
				aspect_ratio_.push_back((data_type)1.f); // default ratio anchor
			}
		}
		else if (config_.valid()) {
			LOG(INFO) << "Using anchor params from config file";

			if (auto_spatial_anchor_) {
				CHECK_GT(inputs_[0]->get(0)->width(), 1);
				CHECK_GT(inputs_[0]->get(0)->height(), 1);

				anchor_num_width_ = inputs_[0]->get(0)->width();
				anchor_num_height_ = inputs_[0]->get(0)->height();
			}
			else {
				anchor_num_width_ = config_.get<int>("spatial_width");
				anchor_num_height_ = config_.get<int>("spatial_height");
			}

			spatial_start_ = config_.get<data_type>("spatial_start");
			spatial_step_ = config_.get<data_type>("spatial_step");

			central_scales_ = config_.get_array<data_type>("central_scale");
			if (central_scales_.empty()) {
				central_scales_.push_back((data_type)1.f); // default scale anchor
			}

			aspect_ratio_ = config_.get_array<data_type>("aspect_ratio");
			if (aspect_ratio_.empty()) {
				aspect_ratio_.push_back((data_type)1.f); // default ratio anchor
			}
		}
		else {
			LOG(FATAL) << "Missing anchor information for anchor_loc_loss_ex_layer";
		}

		// init layer params
		loc_threshold_ = 0.5f;
		total_classes_ = -1;
		loc_coeff_ = (data_type)1.f;
		conf_coeff_ = (data_type)1.f;
		if (layer_param_.has_clsloc_loss_param()) {
			loc_coeff_ = (data_type)(layer_param_.clsloc_loss_param().loc_coeff());
			conf_coeff_ = (data_type)(layer_param_.clsloc_loss_param().cls_coeff());
		}

		assign_reject_iou_ = layer_param_.clsloc_loss_param().assign_reject_iou();
		cls_pos_iou_ = layer_param_.clsloc_loss_param().cls_pos_iou();
		cls_neg_iou_ = layer_param_.clsloc_loss_param().cls_neg_iou();
		prediction_box_classification_ = layer_param_.clsloc_loss_param().prediction_box_classification();
		expected_pos_num_ = layer_param_.clsloc_loss_param().expected_pos_num();
		expected_neg_num_ = layer_param_.clsloc_loss_param().expected_neg_num();

		// init sigmoid layer (for confidence input, inputs_[1])
		sigmoid_bottom_vec_.clear();
		sigmoid_bottom_vec_.push_back(inputs_[1]);
		sigmoid_output_.reset(new node_blob());
		sigmoid_top_vec_.clear();
		sigmoid_top_vec_.push_back(sigmoid_output_);
		sigmoid_layer_.reset(new sigmoid_layer(context_, layer_param_));
		sigmoid_layer_->bind(sigmoid_bottom_vec_, sigmoid_top_vec_);
		sigmoid_layer_->init();

		// setup buffers
		loc_pos_entry_.reset(new node_blob());
		conf_pos_entry_.reset(new node_blob());
		loc_diff_buf_.reset(new node_blob());
		conf_diff_buf_.reset(new node_blob());
		loc_data_buf_.reset(new node_blob());
		conf_data_buf_.reset(new node_blob());
		conf_raw_data_buf_.reset(new node_blob());
	}

	void anchor_loc_loss_ex_layer::resize() {
		check_input();
		sigmoid_layer_->resize();

		int spatial_size = inputs_[0]->get(0)->width() * inputs_[0]->get(0)->height();

		if (auto_spatial_anchor_) {
			CHECK_GT(spatial_size, 1);

			anchor_num_width_ = inputs_[0]->get(0)->width();
			anchor_num_height_ = inputs_[0]->get(0)->height();
		}

		if (inputs_[0]->reshaped()) {
			// prediction input
			// inputs_[0] input format
			// aw * ah * (as * ar * 4 * cls) * n
			// or
			// 1 * 1 * (aw * ah * as * ar * 4 * cls) * n
			if (spatial_size == 1) {
				// format (2)
				int cur_classes = inputs_[0]->get(0)->channels() / (int)aspect_ratio_.size() / (int)central_scales_.size() / anchor_num_width_ / anchor_num_height_ / 4;
				if (total_classes_ < 0) total_classes_ = cur_classes;
				else CHECK_EQ(total_classes_, cur_classes);
				CHECK_EQ(inputs_[0]->get(0)->channels(), aspect_ratio_.size() * central_scales_.size() * anchor_num_width_ * anchor_num_height_ * total_classes_ * 4);
			}
			else {
				CHECK_EQ(inputs_[0]->get(0)->width(), anchor_num_width_);
				CHECK_EQ(inputs_[0]->get(0)->height(), anchor_num_height_);

				int cur_classes = inputs_[0]->get(0)->channels() / (int)aspect_ratio_.size() / (int)central_scales_.size() / 4;
				if (total_classes_ < 0) total_classes_ = cur_classes;
				else CHECK_EQ(total_classes_, cur_classes);
				CHECK_EQ(inputs_[0]->get(0)->channels(), 4 * aspect_ratio_.size() * central_scales_.size() * total_classes_);
			}
		}

		if (inputs_[1]->reshaped()) {
			// confidence input
			// inputs_[1] input format
			// aw * ah * (as * ar * cls) * n
			// or 
			// 1 * 1 * (aw * ah * as * ar * cls) * n
			if (inputs_[1]->get(0)->width() == 1 && inputs_[1]->get(0)->height() == 1) {
				CHECK_EQ(inputs_[1]->get(0)->channels(), total_classes_ * aspect_ratio_.size() * central_scales_.size() * anchor_num_width_ * anchor_num_height_);
			}
			else {
				CHECK_EQ(inputs_[1]->get(0)->width(), anchor_num_width_);
				CHECK_EQ(inputs_[1]->get(0)->height(), anchor_num_height_);
				CHECK_EQ(inputs_[1]->get(0)->channels(), total_classes_ * aspect_ratio_.size() * central_scales_.size());
			}
		}

		if (inputs_[2]->reshaped()) {
			CHECK_EQ(inputs_[2]->get(0)->channels(), 8);
			CHECK_EQ(inputs_[2]->get(0)->height(), 1);
			CHECK_EQ(inputs_[2]->get(0)->width(), 1);
		}

		if (outputs_[0]->size() == 0) {
			outputs_[0]->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, 1, 11, 1, 1, inputs_[0]->get(0)->device_id())
				));

			vector<string> &blob_tags = outputs_[0]->tags();
			blob_tags.resize(11);
			blob_tags[0] = "Anchor LOC Error";
			blob_tags[1] = "Top-Conf LOC Error";
			blob_tags[2] = "LOC Loss";
			blob_tags[3] = "Conf Loss";
			blob_tags[4] = "Pictures With LOC(%)";
			blob_tags[5] = "Pictures Assigned(%)";
			blob_tags[6] = "Pictures Pos(%)";
			blob_tags[7] = "Pictures Neg(%)";
			blob_tags[8] = "Anchors Assigned(%)";
			blob_tags[9] = "Anchors Pos(%)";
			blob_tags[10] = "Anchors Neg(%)";

			LOG(INFO) << "Anchors Width: " << anchor_num_width_;
			LOG(INFO) << "Anchors Height: " << anchor_num_height_;
			LOG(INFO) << "Anchors Scale: " << central_scales_.size();
			LOG(INFO) << "Anchors Ratio: " << aspect_ratio_.size();
			LOG(INFO) << "Total Classes: " << total_classes_;
			LOG(INFO) << "LOC Coeff: " << loc_coeff_;
			LOG(INFO) << "Conf Coeff: " << conf_coeff_;
		}

		int total_anchors = num_anchors();
		int num = inputs_[0]->get(0)->num();
		int device_id = inputs_[0]->get(0)->device_id();

		if (loc_pos_entry_->size() == 0) {
			loc_pos_entry_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, 1, 1, 1, device_id)
				));
			conf_pos_entry_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, 1, 1, 1, device_id)
				));
			loc_diff_buf_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, 4 * total_anchors, 1, 1, device_id)
				));
			conf_diff_buf_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, total_anchors, 1, 1, device_id)
				));
			loc_data_buf_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, 4 * total_anchors, 1, 1, device_id)
				));
			conf_data_buf_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, total_anchors, 1, 1, device_id)
				));
			conf_raw_data_buf_->add(boost::shared_ptr<device_blob>(
				device_blob::create_4d(context_, num, total_anchors, 1, 1, device_id)
				));
		}
		else if (inputs_[0]->reshaped()) {
			loc_pos_entry_->get(0)->reshape_4d(num, 1, 1, 1);
			conf_pos_entry_->get(0)->reshape_4d(num, 1, 1, 1);
			loc_diff_buf_->get(0)->reshape_4d(num, 4 * total_anchors, 1, 1);
			conf_diff_buf_->get(0)->reshape_4d(num, total_anchors, 1, 1);
			loc_data_buf_->get(0)->reshape_4d(num, 4 * total_anchors, 1, 1);
			conf_data_buf_->get(0)->reshape_4d(num, total_anchors, 1, 1);
			conf_raw_data_buf_->get(0)->reshape_4d(num, total_anchors, 1, 1);
		}
	}

	void anchor_loc_loss_ex_layer::on_forward(int device_index) {
		sigmoid_layer_->forward();

		// get data source
		const data_bigfile_layer *data_source = dynamic_cast<const data_bigfile_layer*>(
			context_->get_shared_object(namespace_, SHAREDOBJNAME_DATASOURCE)
			);
		CHECK(data_source) << "Currently only bigfile data sources are supported";
		const vector<data_utils::raw_picture> &metadata = data_source->current_batch().processing_imgs;

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

		const data_type* bottom_loc_data_gpu = inputs_[0]->get(device_index)->gpu_data();
		const data_type* bottom_loc_conf_data_gpu = sigmoid_output_->get(device_index)->gpu_data();
		const data_type* bottom_loc_gt_data = inputs_[2]->get(device_index)->cpu_data();	// use cpu for gt

		data_type* result_data = outputs_[0]->get(device_index)->mutable_cpu_data();

		int num = inputs_[0]->get(device_index)->num();
		int total_anchors = num_anchors();

		int loc_involved = 0;
		int anchor_loc_accuracy = 0;
		int topconf_loc_accuracy = 0;

		correct_.resize(num * total_anchors);

		// prepare cpu buffer
		for (int i = 0; i < num; ++i) {
			int gt_label = metadata[i].label_id;
			loc_pos_entry_->get(device_index)->mutable_cpu_data()[i] = (data_type)(inputs_[0]->get(device_index)->offset(i) + gt_label * 4 * total_anchors);
			conf_pos_entry_->get(device_index)->mutable_cpu_data()[i] = (data_type)(sigmoid_output_->get(device_index)->offset(i) + gt_label * total_anchors);
		}
		cublas.sp2batch(num, total_anchors * 4, bottom_loc_data_gpu, loc_pos_entry_->get(device_index)->gpu_data(), loc_data_buf_->get(device_index)->mutable_gpu_data(), (data_type)0.f);
		cublas.sp2batch(num, total_anchors, bottom_loc_conf_data_gpu, conf_pos_entry_->get(device_index)->gpu_data(), conf_data_buf_->get(device_index)->mutable_gpu_data(), (data_type)0.f);

		for (int i = 0; i < num; ++i) {
			int gt_label = metadata[i].label_id;
			CHECK_LT(gt_label, total_classes_);

			// loc and total accuracy
			box_t<data_type> box_gt_trans(bottom_loc_gt_data + i * 8); // will not be used next

			if (box_gt_trans.valid()) {
				loc_involved++;

				data_utils::box_anchor_transform trans_helper = get_transformer(
					metadata[i].crop_box, metadata[i].width, metadata[i].height);

				data_type max_conf = -FLT_MAX;
				int max_conf_ach = 0;

				// process conf
				for (int ach = 0; ach < total_anchors; ach++) {
					//const data_type *conf_base = bottom_loc_conf_data + bottom[1]->offset(i) + gt_label * total_anchors + ach;
					const data_type *conf_base = conf_data_buf_->get(device_index)->cpu_data() + i * total_anchors + ach;
					data_type v = *conf_base;
					//data_type v = bottom_loc_conf[ach];

					if (v > max_conf) {
						max_conf = v;
						max_conf_ach = ach;
					}
				}

				// process prediction
				for (int ach = 0; ach < total_anchors; ach++) {
					const data_type *predict_base = loc_data_buf_->get(device_index)->cpu_data() + i * total_anchors * 4 + ach;

					data_type predict_buffer[] = {
						predict_base[0], predict_base[total_anchors], predict_base[total_anchors * 2], predict_base[total_anchors * 3]
					};

					box_t<data_type> box_predict_trans;

					if (layer_param_.clsloc_loss_param().loss_transform() == ClsLocLossParameter_LossTransform_LTRB) {
						box_predict_trans = box_t<data_type>(predict_buffer);
					}
					else if (layer_param_.clsloc_loss_param().loss_transform() == ClsLocLossParameter_LossTransform_CX_CY_LOGW_LOGH) {
						box_predict_trans = box_t<data_type>::from_CX_CY_LOGW_LOGH(predict_buffer);
					}
					else {
						LOG(FATAL) << "Unknown loss_transform type: " << (int)layer_param_.clsloc_loss_param().loss_transform();
					}

					Box box_predict = trans_helper.transform_back(box_predict_trans, ach, true);

					double max_iou = 0;
					for (auto iter = metadata[i].bounding_boxes.begin(); iter != metadata[i].bounding_boxes.end(); ++iter) {
						double iou = box_predict.IoU(*iter);
						max_iou = std::max(max_iou, iou);
					}

					if (max_iou > loc_threshold_) {
						anchor_loc_accuracy++;

						if (max_conf_ach == ach) {
							topconf_loc_accuracy++;
						}

						correct_[i * total_anchors + ach] = 1;
					}
					else {
						correct_[i * total_anchors + ach] = 0;
					}
				}
			}
		}

		result_data[0] = 1 - (loc_involved == 0 ? 0 : (data_type)anchor_loc_accuracy / (loc_involved * total_anchors));
		result_data[1] = 1 - (loc_involved == 0 ? 0 : (data_type)topconf_loc_accuracy / loc_involved);

		result_data[4] = (data_type)loc_involved / num * 100;
	}

	void anchor_loc_loss_ex_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		
		if (!should_bp(bp_acts, 0) && !should_bp(bp_acts, 1)) {
			return;
		}

		// get data source
		const data_bigfile_layer *data_source = dynamic_cast<const data_bigfile_layer*>(
			context_->get_shared_object(namespace_, SHAREDOBJNAME_DATASOURCE)
			);
		CHECK(data_source) << "Currently only bigfile data sources are supported";
		const vector<data_utils::raw_picture> &metadata = data_source->current_batch().processing_imgs;

		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		// bottom[0]: predict
		// bottom[1]: confidence
		// bottom[2]: gt 

		data_type* loc_diff_gpu = inputs_[0]->get(device_index)->mutable_gpu_diff();
		data_type* conf_diff_gpu = inputs_[1]->get(device_index)->mutable_gpu_diff();

		data_type loc_loss = 0;
		int loc_involved = 0;
		int loc_assigned = 0;
		int anchor_assigned = 0;
		const data_type *loc_data_gpu = inputs_[0]->get(device_index)->gpu_data();
		const data_type *loc_gt_data = inputs_[2]->get(device_index)->cpu_data();	// use cpu for gt

		data_type conf_loss = 0;
		int cls_pos = 0;
		int cls_neg = 0;
		int anchor_cls_pos = 0;
		int anchor_cls_neg = 0;

		const data_type *conf_data_gpu = sigmoid_output_->get(device_index)->gpu_data(); // not inputs_[1]!
		const data_type *conf_raw_data_gpu = inputs_[1]->get(device_index)->gpu_data();

		data_type beta_predict = get_beta(clear_acts_diff, 0);
		data_type beta_confidence = get_beta(clear_acts_diff, 1);

		if (should_bp(bp_acts, 0) && beta_predict == 0) {
			CUDA_CHECK(cudaMemset(loc_diff_gpu, 0, sizeof(data_type)* inputs_[0]->get(device_index)->count()));
		}
		if (should_bp(bp_acts, 1) && beta_confidence == 0) {
			CUDA_CHECK(cudaMemset(conf_diff_gpu, 0, sizeof(data_type)* inputs_[1]->get(device_index)->count()));
		}

		int num = inputs_[0]->get(device_index)->num();
		int total_anchors = num_anchors();

		// prepare cpu buffers
		// here, loc_pos_entry_ and conf_pos_entry is ready in Forward_gpu
		// loc_data_buf_ and conf_data_buf_ is also ready
		memset(loc_diff_buf_->get(device_index)->mutable_cpu_data(), 0, loc_diff_buf_->get(device_index)->count() * sizeof(data_type));
		memset(conf_diff_buf_->get(device_index)->mutable_cpu_data(), 0, conf_diff_buf_->get(device_index)->count() * sizeof(data_type));
		cublas.sp2batch(num, total_anchors, conf_raw_data_gpu, conf_pos_entry_->get(device_index)->gpu_data(), conf_raw_data_buf_->get(device_index)->mutable_gpu_data(), (data_type)0.f);

		for (int i = 0; i < num; i++) {
			int cls_label = metadata[i].label_id;

			const data_type* cur_gt_data = loc_gt_data + i * 8; // will not used next
			box_t<data_type> cur_gt_box_trans(cur_gt_data);

			if (cur_gt_box_trans.valid()) {
				loc_involved++;

				data_utils::box_anchor_transform trans_helper = get_transformer(metadata[i].crop_box, metadata[i].width, metadata[i].height);

				// select anchors
				vector<int> loc_anchors, pos_anchors, neg_anchors;

				for (int ach = 0; ach < total_anchors; ach++) {
					data_type assigned_iou;
					trans_helper.get_maxiou_box(ach, metadata[i].bounding_boxes, &assigned_iou);

					if (assigned_iou >= assign_reject_iou_) loc_anchors.push_back(ach);
					if (assigned_iou >= cls_pos_iou_) pos_anchors.push_back(ach);
					if (assigned_iou < cls_neg_iou_) neg_anchors.push_back(ach);
				}


				if (expected_pos_num_ >= 0) {
					std::random_shuffle(loc_anchors.begin(), loc_anchors.end());
					std::random_shuffle(pos_anchors.begin(), pos_anchors.end());
					while (loc_anchors.size() > expected_pos_num_) loc_anchors.pop_back();
					while (pos_anchors.size() > expected_pos_num_) pos_anchors.pop_back();
				}

				if (expected_neg_num_ >= 0) {
					std::random_shuffle(neg_anchors.begin(), neg_anchors.end());
					while (neg_anchors.size() > expected_neg_num_) neg_anchors.pop_back();
				}

				int cur_loc_assigned = 0;
				int cur_cls_pos = 0;
				int cur_cls_neg = 0;

				// loc term
				for (vector<int>::iterator iter = loc_anchors.begin(); iter != loc_anchors.end(); ++iter) {
					int ach = *iter;

					//const data_type *predict_base = &loc_data_cpu[ach];
					//data_type *predict_diff_base = &loc_diff_cpu[ach];

					const data_type *predict_base = loc_data_buf_->get(device_index)->cpu_data() + i * total_anchors * 4 + ach;
					data_type *predict_diff_base = loc_diff_buf_->get(device_index)->mutable_cpu_data() + i * total_anchors * 4 + ach;

					Box assigned_gt_box = trans_helper.get_maxiou_box(ach, metadata[i].bounding_boxes);

					cur_loc_assigned = 1;
					anchor_assigned++;
					box_t<data_type> trans_assigend_gt_box = trans_helper.transform(assigned_gt_box, ach);

					// loss_transform would never applied on cur_gt_box_trans
					// so transform it to gt[] here
					data_type gt[4];

					if (layer_param_.clsloc_loss_param().loss_transform() == ClsLocLossParameter_LossTransform_LTRB) {
						trans_assigend_gt_box.fill(gt);
					}
					else if (layer_param_.clsloc_loss_param().loss_transform() == ClsLocLossParameter_LossTransform_CX_CY_LOGW_LOGH) {
						trans_assigend_gt_box.fill_CX_CY_LOGW_LOGH(gt);
					}
					else {
						LOG(FATAL) << "Unknown loss_transform type: " << (int)layer_param_.clsloc_loss_param().loss_transform();
					}

					for (int j = 0; j < 4; j++) {
						data_type v = (predict_base[j * total_anchors] - gt[j]);

						if (v > 1 || v < -1) { // l1 loss
							loc_loss += abs(v);
							predict_diff_base[j * total_anchors] = v > 0 ? (data_type)1.f : (data_type)-1.f;
						}
						else { // l2 loss
							loc_loss += v * v;
							predict_diff_base[j * total_anchors] = v;
						}
					}
				}

				// pos conf term
				for (vector<int>::iterator iter = pos_anchors.begin(); iter != pos_anchors.end(); ++iter) {
					int ach = *iter;

					const data_type *conf_base = conf_data_buf_->get(device_index)->cpu_data() + i * total_anchors + ach;
					const data_type *conf_raw_base = conf_raw_data_buf_->get(device_index)->cpu_data() + i * total_anchors + ach;
					data_type *conf_diff_base = conf_diff_buf_->get(device_index)->mutable_cpu_data() + i * total_anchors + ach;

					int is_correct = 1;
					cur_cls_pos = 1;
					anchor_cls_pos++;

					conf_diff_base[0] = conf_base[0] - is_correct;
					conf_loss -= conf_raw_base[0] * (is_correct - (conf_raw_base[0] >= 0)) -
						log(1 + exp(conf_raw_base[0] - 2 * conf_raw_base[0] * (conf_raw_base[0] >= 0)));
				}

				// neg conf term
				for (vector<int>::iterator iter = neg_anchors.begin(); iter != neg_anchors.end(); ++iter) {
					int ach = *iter;

					const data_type *conf_base = conf_data_buf_->get(device_index)->cpu_data() + i * total_anchors + ach;
					const data_type *conf_raw_base = conf_raw_data_buf_->get(device_index)->cpu_data() + i * total_anchors + ach;
					data_type *conf_diff_base = conf_diff_buf_->get(device_index)->mutable_cpu_data() + i * total_anchors + ach;

					int is_correct = 0;
					cur_cls_neg = 1;
					anchor_cls_neg++;

					conf_diff_base[0] = conf_base[0] - is_correct;
					conf_loss -= conf_raw_base[0] * (is_correct - (conf_raw_base[0] >= 0)) -
						log(1 + exp(conf_raw_base[0] - 2 * conf_raw_base[0] * (conf_raw_base[0] >= 0)));
				}

				loc_assigned += cur_loc_assigned;
				cls_pos += cur_cls_pos;
				cls_neg += cur_cls_neg;
			}
		}

		if (anchor_assigned > 0) {
			cublas.scal(loc_diff_buf_->get(device_index)->count(), loc_coeff_ / anchor_assigned, loc_diff_buf_->get(device_index)->mutable_gpu_data());
			loc_loss /= anchor_assigned * 2;
		}

		if (anchor_cls_pos + anchor_cls_neg > 0) {
			cublas.scal(conf_diff_buf_->get(device_index)->count(), conf_coeff_ / (anchor_cls_pos + anchor_cls_neg), conf_diff_buf_->get(device_index)->mutable_gpu_data());
			conf_loss /= anchor_cls_pos + anchor_cls_neg;
		}

		if (should_bp(bp_acts, 0)) {
			cublas.batch2sp(num, total_anchors * 4, loc_diff_buf_->get(device_index)->gpu_data(), loc_pos_entry_->get(device_index)->gpu_data(), loc_diff_gpu, beta_predict);
		}
		if (should_bp(bp_acts, 1)) {
			cublas.batch2sp(num, total_anchors, conf_diff_buf_->get(device_index)->gpu_data(), conf_pos_entry_->get(device_index)->gpu_data(), conf_diff_gpu, beta_confidence);
		}

		data_type* result_data = outputs_[0]->get(device_index)->mutable_cpu_data();
		result_data[2] = loc_loss;
		result_data[3] = conf_loss;

		result_data[5] = (data_type)loc_assigned / num * 100;
		result_data[6] = (data_type)cls_pos / num * 100;
		result_data[7] = (data_type)cls_neg / num * 100;
		result_data[8] = (data_type)anchor_assigned / (total_anchors * num) * 100;
		result_data[9] = (data_type)anchor_cls_pos / (total_anchors * num) * 100;
		result_data[10] = (data_type)anchor_cls_neg / (total_anchors * num) * 100;
	}

	data_utils::box_anchor_transform anchor_loc_loss_ex_layer::get_transformer(const data_utils::Box &base_box, int im_width, int im_height) {
		return data_utils::box_anchor_transform(
			im_width, im_height, 
			anchor_num_width_, anchor_num_height_, 
			base_box, 
			central_scales_, aspect_ratio_, spatial_start_, spatial_step_
			);
	}
}