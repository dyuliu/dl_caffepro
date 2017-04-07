
#pragma once

#include <caffepro/object_model/caffepro_net.h>

namespace caffepro {

	class bn_postprocess {
	public:
		bn_postprocess(caffepro_context *context, const std::string &net_def_file, const std::string &net_binary_file, const analyzer_proto::Info &info);
		bn_postprocess(caffepro_context *context, const std::string &net_def_file, const std::string &net_binary_file);
		bn_postprocess(caffepro_context *context, NetParameter &net_def, NetParameter &net_binary);

	public:
		// interfaces
		void run(int turn);
		void save(NetParameter &net_binary);
		void save(const std::string &filename);

	public:
		// fetch functions
		caffepro_context *context() const { return context_; }
		boost::shared_ptr<caffepro_net> net(){ return net_; }

	protected:
		void init(NetParameter &net_def, NetParameter &net_binary);

	protected:
		caffepro_context *context_;
		boost::shared_ptr<caffepro_net> net_;
	};
}