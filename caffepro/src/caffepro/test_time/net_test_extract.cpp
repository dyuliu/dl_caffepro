
#include <caffepro/test_time/net_test_extract.h>
#include <caffepro/utils/color_print.h>
#include <caffepro/utils/filesystem.h>

namespace caffepro {
	using std::string;

	net_test_extract::net_test_extract(boost::shared_ptr<caffepro_net> net)
		: net_(net) {
		// nothing to do
	}

	static void save_layer_data_filename(std::string tgt_fold, std::string filename, boost::shared_ptr<device_blob> &blob) {
		filesystem::create_directory(tgt_fold.data());
		std::ofstream fp(tgt_fold+"//"+filename+".info", std::ios::out);
		blob->save_data(fp);
		fp.close();
	}

	void net_test_extract::run(int num_iters, std::string layer_name) {
		CHECK_GE(num_iters, 0);

		COUT_METD << "Start to extract feature: " << layer_name << std::endl;
		std::string layer_name_ = layer_name;
		auto layer_data_ = net_->get_layer(layer_name_)->outputs()[0]->get(0);

		net_->context()->set_phase(caffepro_context::TEST);

		int during_time = 0;
		for (int iter = 0; iter < num_iters; iter++) {
			clock_t start_time = clock();
			net_->forward(false);
			auto ptr = net_->data_provider()->current_batch()->batch_data[0].original_data.get()->data_name;
			net_->context()->sync_all_devices();
			save_layer_data_filename(layer_name_, ptr,layer_data_);
			int forward_time = clock() - start_time;
			during_time += forward_time;
			COUT_SUCC << "Processing the images: " << iter << " Spend time: " << forward_time << std::endl;
		}
		COUT_SUCC << "Total sample " << num_iters << ", spend time " << during_time/60.0 << " s" << std::endl;
	}
}