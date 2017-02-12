
#pragma once

#include <caffepro/object_model/caffepro_net.h>

namespace caffepro {
	class net_tester : public caffepro_object {
	public:
		// definition
		struct metric {
			std::string source;
			std::string name;
			double value;
		};

	public:
		net_tester(boost::shared_ptr<caffepro_net> net);

	public:
		// interfaces
		void run(int num_iters, int display_iters, const std::string &info_file);

	public:
		// fetch functions
		boost::shared_ptr<caffepro_net> net(){ return net_; }

	protected:
		virtual void merge_metrics(std::vector<metric> &metrics, caffepro_net &net);
		virtual void display_metrics(int cur_iter, std::vector<metric> &metrics, const std::string prefix);

	protected:
		boost::shared_ptr<caffepro_net> net_;
	};
}