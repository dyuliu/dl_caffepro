
#include <caffepro/utils/filler.h>

namespace caffepro {

	// factory function
	filler* filler::create(caffepro_context *context, const FillerParameter& param) {
		const std::string &type = param.type();

		if (type == "constant") {
			return new constant_filler(context, param);
		}
		else if (type == "gaussian") {
			return new gaussian_filler(context, param);
		}
		else if (type == "uniform") {
			return new uniform_filler(context, param);
		}
		else if (type == "xavier") {
			return new xavier_filler(context, param);
		}
		else if (type == "xiangyu") {
			return new xiangyu_filler(context, param);
		}
		else if (type == "identity") {
			return new identity_filler(context, param);
		}
		else {
			LOG(FATAL) << "Unknown filler name : " << param.type();
		}

		return nullptr;
	}
}