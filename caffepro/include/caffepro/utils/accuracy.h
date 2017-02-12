
#pragma once 

#include <caffepro/object_model/device_blob.h>

namespace caffepro {
	void compute_multilabel_accurancy(const device_blob *prob, const device_blob *label, 
		__out float *accuracy, __out float *loss);
}