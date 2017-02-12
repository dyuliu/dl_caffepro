
#include <caffepro/layers/vision_layers.h>
#include <caffepro/proto/caffe.pb.h>

namespace caffepro {
	caffepro_layer* caffepro_layer::create(caffepro_context *context, const LayerParameter &param) {
		if (param.type() == "conv") {
			return new conv_layer(context, param);
		}
		else if (param.type() == "inner_product") {
			return new innerproduct_layer(context, param);
		}
		else if (param.type() == "pool") {
			return new pooling_layer(context, param);
		}
		else if (param.type() == "relu") {
			return new relu_layer(context, param);
		}
		else if (param.type() == "prelu") {
			return new prelu_layer(context, param);
		}
		else if (param.type() == "gpu_concat") {
			return new gpu_concat_layer(context, param);
		}
		else if (param.type() == "softmax") {
			return new softmax_layer(context, param);
		}
		else if (param.type() == "softmax_loss") {
			return new softmax_loss_layer(context, param);
		}
		else if (param.type() == "dropout") {
			return new dropout_layer(context, param);
		}
		else if (param.type() == "dropoutsame") {
			return new dropoutsame_layer(context, param);
		}
		else if (param.type() == "eltwise_sum" || param.type() == "dumb") {
			return new eltwise_sum_layer(context, param);
		}
		else if (param.type() == "eltwise_prod" || param.type() == "eltwise_product") {
			return new eltwise_prod_layer(context, param);
		}
		else if (param.type() == "eltwise_max") {
			return new eltwise_max_layer(context, param);
		}
		else if (param.type() == "eltwise_amax" || param.type() == "eltwise_absmax") {
			return new eltwise_amax_layer(context, param);
		}
		else if (param.type() == "data_bigfile") {
			return new data_bigfile_layer(context, param);
		}
		else if (param.type() == "batch_norm") {
			return new batch_norm_layer(context, param);
		}
		else if (param.type() == "runavg_batch_norm") {
			return new runavg_batch_norm_layer(context, param);
		}
		else if (param.type() == "mean_norm") {
			return new mean_norm_layer(context, param);
		}
		else if (param.type() == "concat") {
			return new concat_layer(context, param);
		}
		else if (param.type() == "sigmoid") {
			return new sigmoid_layer(context, param);
		}
		else if (param.type() == "multilabel_sigmoid_cross_entropy_loss") {
			return new multilabel_sigmoid_loss_layer(context, param);
		}
		else if (param.type() == "black_hole") {
			return new black_hole_layer(context, param);
		}
		else if (param.type() == "anchorloc_ex_loss") {
			return new anchor_loc_loss_ex_layer(context, param);
		}
		else if (param.type() == "grid_generator") {
			return new grid_generator_layer(context, param);
		}
		else if (param.type() == "sample") {
			return new sample_layer(context, param);
		}
		else if (param.type() == "data_entry") {
			return new data_entry_layer(context, param);
		}
		else if (param.type() == "rcnn_loss") {
			return new rcnn_loss_layer(context, param);
		}
		else if (param.type() == "box_pool") {
			return new box_pool_layer(context, param);
		}
		else if (param.type() == "box_regression_loss") {
			return new box_regression_layer(context, param);
		}
		else if (param.type() == "resize_grid") {
			return new resize_grid_layer(context, param);
		}
		else if (param.type() == "exp") {
			return new exp_layer(context, param);
		}
		else if (param.type() == "observer") {
			return new observer_layer(context, param);
		}
		else if (param.type() == "bound") {
			return new bound_layer(context, param);
		}
		else if (param.type() == "cluster_loss") {
			return new cluster_loss_layer(context, param);
		}
		else if (param.type() == "lrn") {
			return new lrn_layer(context, param);
		}
		else if (param.type() == "local_norm") {
			return new local_norm_layer(context, param);
		}
		else if (param.type() == "online_kmeans_loss") {
			return new online_kmeans_loss_layer(context, param);
		}
		else if (param.type() == "padding") {
			return new padding_layer(context, param);
		}
		else if (param.type() == "weight") {
			return new weight_layer(context, param);
		}
		else if (param.type() == "correlation") {
			return new correlation_layer(context, param);
		}
		else if (param.type() == "scalebias") {
			return new scalebias_layer(context, param);
		}
		else if (param.type() == "crop") {
			return new crop_layer(context, param);
		}
		else if (param.type() == "accuracy") {
			return new accuracy_layer(context, param);
		}
		else if (param.type() == "flip") {
			return new flip_layer(context, param);
		}
		else if (param.type() == "reduce_dim") {
			return new reduce_dim_layer(context, param);
		}
		else if (param.type() == "step_gate") {
			return new step_gate_layer(context, param);
		}
		else if (param.type() == "euclidean_loss") {
			return new euclidean_loss_layer(context, param);
		}
		else if (param.type() == "softthr") {
			return new softthr_layer(context, param);
		}
		else if (param.type() == "l2norm") {
			return new l2norm_layer(context, param);
		}
		else if (param.type() == "eltwise_max") {
			return new eltwise_max_layer(context, param);
		}
		else if (param.type() == "eltwise_amax") {
			return new eltwise_amax_layer(context, param);
		}
		else if (param.type() == "householder") {
			return new householder_layer(context, param);
		}
		else if (param.type() == "instance_sample") {
			return new instance_sample_layer(context, param);
		}
		else if (param.type() == "matrix_mul") {
			return new matrix_mul_layer(context, param);
		}
		else if (param.type() == "matrix_mul_stack") {
			return new matrix_mul_stack_layer(context, param);
		}
		else if (param.type() == "transpose") {
			return new transpose_layer(context, param);
		}
		else if (param.type() == "transpose4d") {
			return new transpose4d_layer(context, param);
		}
		else if (param.type() == "learnable_dropout") {
			return new learnable_dropout_layer(context, param);
		}
		else if (param.type() == "dim_innerproduct") {
			return new dim_innerproduct_layer(context, param);
		}
		else if (param.type() == "sym_sigmoid") {
			return new sym_sigmoid_layer(context, param);
		}
		else if (param.type() == "rand_select") {
			return new rand_select_layer(context, param);
		}
		else if (param.type() == "l1norm") {
			return new l1norm_layer(context, param);
		}
		else if (param.type() == "slice") {
			return new slice_layer(context, param);
		}
		else if (param.type() == "dimshuffle") {
			return new dimshuffle_layer(context, param);
		}
		else if (param.type() == "diag_operation") {
			return new diag_operation_layer(context, param);
		}
		else if (param.type() == "diag4d_operation") {
			return new diag4d_operation_layer(context, param);
		}
		else if (param.type() == "birelu") {
			return new birelu_layer(context, param);
		}
		else if (param.type() == "softmax_loss_ohem") {
			return new softmax_loss_ohem_layer(context, param);
		}
		else if (param.type() == "softmax_ohem") {
			return new softmax_ohem_layer(context, param);
		}
		else if (param.type() == "softmax_ohem_split") {
			return new softmax_ohem_split_layer(context, param);
		}
		else {
			LOG(FATAL) << "Unknown layer name: " << param.type();
		}

		return nullptr;
	}
}