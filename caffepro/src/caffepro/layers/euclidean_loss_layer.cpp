
#include <caffepro/layers/euclidean_loss_layer.h>
#include <caffepro/proto/caffe.pb.h>
#include <caffepro/math/cublas_wrapper.h>
#include <caffepro/utils/utils.h>
#include <caffepro/context/common_names.h>
#include <caffepro/utils/string_uitls.h>

#include <functional>

#define MAX_MINIBATCH 512

namespace caffepro {
	euclidean_loss_layer::euclidean_loss_layer(caffepro_context *context, const LayerParameter &param)
		: caffepro_layer(context, param) {

		attr_.num_inputs_min = attr_.num_inputs_max = 1;
		attr_.num_outputs_min = attr_.num_outputs_max = 1;

		attr_.set_constraint(
			layer_attribute::CF_REQUIRE_NDIM_4
			| layer_attribute::CF_REQUIRE_FIXEDLEN_DIM // remove it in the future
			);

		attr_.usage = layer_attribute::USAGE_LOSS;
	}

	euclidean_loss_layer::~euclidean_loss_layer() {
		release_all();
	}

	void euclidean_loss_layer::init() {
		check_input();

		coeff_ = layer_param_.loss_param().coeff();

		sum_multiplier_.reset(new node_blob());
		avg_loss_.reset(new node_blob());
	}

	void euclidean_loss_layer::resize() {
		check_input();
		
		int n_devices = (int)inputs_[0]->size();
		if (outputs_[0]->size() == 0) {

			int num_of_outputs = 2;
			outputs_[0]->tags().resize(num_of_outputs);
			outputs_[0]->tags()[0] = "PSNR";
			outputs_[0]->tags()[1] = "Loss"; // L2 loss

			if (!layer_param_.loss_param().display_result()) {
				outputs_[0]->tags().clear();
			}
			
			for (int nd = 0; nd < n_devices; nd++) {
				outputs_[0]->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(context_, 1, num_of_outputs, 1, 1, inputs_[0]->get(nd)->device_id())
					));

				avg_loss_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
					context_,
					inputs_[0]->get(nd)->num(),
					1,
					1,
					1,
					inputs_[0]->get(nd)->device_id()
					)
					));

				sum_multiplier_->add(boost::shared_ptr<device_blob>(
					device_blob::create_4d(
					context_,
					1,
					inputs_[0]->get(nd)->channels(),
					inputs_[0]->get(nd)->height(),
					inputs_[0]->get(nd)->width(),
					inputs_[0]->get(nd)->device_id()
					)
					));
				sum_multiplier_->get(nd)->fill_data((data_type)1.f);
			}
		}
		else if (inputs_[0]->reshaped()) {
			// do not need to reshape top

			for (int nd = 0; nd < n_devices; nd++) {
				avg_loss_->get(nd)->reshape_4d(
					inputs_[0]->get(nd)->num(),
					1,
					1,
					1
					);

				sum_multiplier_->get(nd)->reshape_4d(1, inputs_[0]->get(nd)->channels(), inputs_[0]->get(nd)->height(), inputs_[0]->get(nd)->width());
				sum_multiplier_->get(nd)->fill_data((data_type)1.f);
			}
		}
	}

	void euclidean_loss_layer::on_forward(int device_index) {
		
		const int num = inputs_[0]->get(device_index)->num();
		const int channels = inputs_[0]->get(device_index)->channels();
		const int spatial_size = inputs_[0]->get(device_index)->width() * inputs_[0]->get(device_index)->height();
		const int count = inputs_[0]->get(device_index)->count();
		
		cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());
		
		data_type* top_data = reinterpret_cast<data_type*>(context_->get_current_device()->memory()->allocate(count * sizeof(data_type)));

		// compute dot prod
		cublas.mul(count, inputs_[0]->get(device_index)->gpu_data(), inputs_[0]->get(device_index)->gpu_data(), top_data); // x^2

		cublas.gemv(
			CblasNoTrans,
			num, spatial_size * channels,
			(data_type)1. / (data_type)(spatial_size * channels),
			top_data,
			sum_multiplier_->get(device_index)->gpu_data(),
			(data_type)0.,
			avg_loss_->get(device_index)->mutable_gpu_data()
			);

		// psnr
		data_type avg_loss = 0.0f;
		data_type avg_psnr = 0.0f;
		for (int i = 0; i < num; i++) {
			//data_type mse = avg_loss_->get(device_index)->cpu_data()[i] / 255.0f / 255.0f;
			data_type mse = avg_loss_->get(device_index)->cpu_data()[i];
			data_type psnr = -data_type(10) * log(mse) / log(data_type(10));
			//printf("i: %d, psnr: %.5f\n", i, psnr);
			
			avg_psnr += psnr / (data_type)num;
			avg_loss += avg_loss_->get(device_index)->cpu_data()[i] / (data_type)num;
		}
		outputs_[0]->get(device_index)->mutable_cpu_data()[0] = avg_psnr; 
		outputs_[0]->get(device_index)->mutable_cpu_data()[1] = avg_loss;

		//std::cout << "output[0]: " << outputs_[0]->get(device_index)->cpu_data()[0] << std::endl;
		//std::cout << "output[1]: " << outputs_[0]->get(device_index)->cpu_data()[1] << std::endl;
		
		context_->get_current_device()->memory()->free(top_data);
	}

	void euclidean_loss_layer::on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) {
		const data_type beta_acts = get_beta(clear_acts_diff, 0);

		if (should_bp(bp_acts, 0)) {
			cublas_wrapper<data_type> cublas(context_, context_->get_current_device()->device_id());

			const int count = inputs_[0]->get(device_index)->count();
			cublas.axpby(count,
				data_type(2.0f * coeff_ / count), inputs_[0]->get(device_index)->gpu_data(),
				beta_acts, inputs_[0]->get(device_index)->mutable_gpu_diff());
		}
	}
}