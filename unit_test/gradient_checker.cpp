
#include <gradient_checker.h>
#include <iostream>

using std::cout;
using std::endl;

namespace caffepro {

	gradient_checker::gradient_checker(caffepro_layer::layer_io_buffer &inputs, caffepro_layer::layer_io_buffer &outputs)
		: inputs_(inputs), outputs_(outputs) {

		// random init input
		FillerParameter fill_param;
		fill_param.set_type("gaussian");
		fill_param.set_mean(3.f);
		fill_param.set_std(5.f);
		CHECK_GT(inputs.size(), 0);
		for (auto pI = inputs.begin(); pI != inputs.end(); ++pI) {
			CHECK_GT((*pI)->size(), 0);
			boost::shared_ptr<filler> fil(filler::create((*pI)->get(0)->context(), fill_param));
			fil->fill(*(*pI)->get(0));
			//break;
		}

		// fill output diff to 1
		CHECK_GE(outputs.size(), 1);
		
		CHECK_GT(outputs[0]->size(), 0);
		outputs[0]->get(0)->fill_diff(1.f);

		for (int i = 1; i < outputs.size(); i++){
			CHECK_GT(outputs[i]->size(), 0);
			outputs[i]->get(0)->fill_diff(0.f);
		}
	}

	void gradient_checker::check_layer(caffepro_layer &layer) {

		// forward and backward
		layer.forward();

		layer.context()->sync_all_devices();
		double original_target = outputs_[0]->get(0)->sum();
		//double original_target = outputs_[0]->get(0)->cpu_data()[0];

		layer.backward();
		layer.context()->sync_all_devices();

		
		//double original_target = outputs_[0]->get(0)->cpu_data()[1];

		// eval act gradient
		for (int i = 0; i < (int)inputs_.size(); i++) {
			double input_mean = inputs_[i]->get(0)->mean();
			double input_var = inputs_[i]->get(0)->variance();
			double delta = sqrt(input_var) * 1e-1;
			cout << "inputs[" << i << "]: mean: " << input_mean << " var: " << input_var << " delta: " << delta << endl;

			double mean_grad = 0, mean_grad_delta = 0;
			int count = (int)inputs_[i]->get(0)->count();

			for (int j = 0; j < count; j++) {
				data_type *data = inputs_[i]->get(0)->mutable_cpu_data();
				float old_data = data[j];

				data[j] += (data_type)delta;

				layer.forward();
				layer.context()->sync_all_devices();

				double new_target = outputs_[0]->get(0)->sum();
				//double new_target = outputs_[0]->get(0)->cpu_data()[0];
				double simu_gradient = (new_target - original_target) / delta;
				double bp_gradient = inputs_[i]->get(0)->cpu_diff()[j];

				mean_grad += abs(bp_gradient);
				mean_grad_delta += abs(simu_gradient - bp_gradient);

				// recover
				inputs_[i]->get(0)->mutable_cpu_data()[j] = old_data;
			}

			cout << "mean_grad: " << mean_grad / count << " grad_delta: " << mean_grad_delta / count << endl << endl;
		}

		// eval weight gradient
		for (int i = 0; i < (int)layer.weights().size(); i++) {
			double weight_mean = layer.weights()[i]->get(0)->mean();
			double weight_var = layer.weights()[i]->get(0)->variance();
			double mean2 = layer.weights()[i]->get(0)->mean2();
			double delta = sqrt(mean2) * 1e-1;
			if (delta == 0) delta = 1e-4;
			cout << "weights[" << i << "]: mean: " << weight_mean << " var: " << weight_var << " delta: " << delta << endl;

			double mean_grad = 0, mean_grad_delta = 0;
			int count = (int)layer.weights()[i]->get(0)->count();
			for (int j = 0; j < count; j++) {
				data_type *data = layer.weights()[i]->get(0)->mutable_cpu_data();
				float old_data = data[j];
				data[j] += (data_type)delta;

				layer.forward();
				layer.context()->sync_all_devices();

				double new_target = outputs_[0]->get(0)->sum();
				//double new_target = outputs_[0]->get(0)->cpu_data()[0];
				double simu_gradient = (new_target - original_target) / delta;
				double bp_gradient = layer.weights()[i]->get(0)->cpu_diff()[j];

				mean_grad += abs(bp_gradient);
				mean_grad_delta += abs(simu_gradient - bp_gradient);

				// recover
				layer.weights()[i]->get(0)->mutable_cpu_data()[j] = old_data;
			}

			cout << "mean_grad: " << mean_grad / count << " grad_delta: " << mean_grad_delta / count << endl << endl;
		}
	}

	void gradient_checker::check_net(caffepro_net &net) {

		net.setup_layer_runtime_properties(true);

		// forward and backward
		net.forward();
		net.backward();

		net.context()->sync_all_devices();
		double original_target = outputs_[0]->get(0)->sum();
		//double original_target = outputs_[0]->get(0)->cpu_data()[1];

		// eval act gradient
		for (int i = 0; i < (int)inputs_.size(); i++) {
			double input_mean = inputs_[i]->get(0)->mean();
			double input_var = inputs_[i]->get(0)->variance();
			double delta = sqrt(input_var) * 1e-1;
			cout << "inputs[" << i << "]: mean: " << input_mean << " var: " << input_var << " delta: " << delta << endl;

			double mean_grad = 0, mean_grad_delta = 0;
			int count = (int)inputs_[i]->get(0)->count();

			for (int j = 0; j < count; j++) {
				data_type *data = inputs_[i]->get(0)->mutable_cpu_data();
				float old_data = data[j];

				data[j] += (data_type)delta;

				net.forward();
				net.context()->sync_all_devices();

				double new_target = outputs_[0]->get(0)->sum();
				//double new_target = outputs_[0]->get(0)->cpu_data()[1];
				double simu_gradient = (new_target - original_target) / delta;
				double bp_gradient = inputs_[i]->get(0)->cpu_diff()[j];

				mean_grad += abs(bp_gradient);
				mean_grad_delta += abs(simu_gradient - bp_gradient);

				// recover
				inputs_[i]->get(0)->mutable_cpu_data()[j] = old_data;
			}

			cout << "mean_grad: " << mean_grad / count << " grad_delta: " << mean_grad_delta / count << endl << endl;
		}
	}
}