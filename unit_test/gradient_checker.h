
#pragma once

#include <caffepro/object_model/caffepro_layer.h>
#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/utils/filler.h>

namespace caffepro {
	class gradient_checker {
	public:
		gradient_checker(caffepro_layer::layer_io_buffer &inputs, caffepro_layer::layer_io_buffer &outputs);

		void check_layer(caffepro_layer &layer);
		void check_net(caffepro_net &net);

	private:
		caffepro_layer::layer_io_buffer &inputs_, &outputs_;
	};
}