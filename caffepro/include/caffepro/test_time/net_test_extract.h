
#pragma once

#include <caffepro/object_model/caffepro_net.h>

namespace caffepro {
	class net_test_extract : public caffepro_object {

	public:
		net_test_extract(boost::shared_ptr<caffepro_net> net);

	public:
		// interfaces
		void run(int num_iters, std::string layer_name);

	public:
		// fetch functions
		boost::shared_ptr<caffepro_net> net() { return net_; }

	protected:
		boost::shared_ptr<caffepro_net> net_;
	};
}