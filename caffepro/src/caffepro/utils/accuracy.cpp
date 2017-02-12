
#include <caffepro/utils/accuracy.h>
#include <caffepro/caffepro.h>
#include <set>

namespace caffepro {
	void compute_multilabel_accurancy(const device_blob *prob, const device_blob *label, 
		__out float *accuracy, __out float *loss) {
		
		*accuracy = 0;

		CHECK(prob->same_shape(*label));
		const data_type* bottom_data = prob->cpu_data();
		const data_type* bottom_label = label->cpu_data();
		int num = prob->num();
		int dim = prob->count() / prob->num();

		for (int i = 0; i < num; ++i) {
			// Multi-label Accuracy

			const data_type* cur_bottom_data = bottom_data + i * dim;
			const data_type* cur_bottom_label = bottom_label + i * dim;

			std::vector<std::pair<data_type, int> > score(dim);
			for (int j = 0; j < dim; j++) {
				score[j] = std::make_pair(cur_bottom_data[j], j);
			}
			sort(score.begin(), score.end());

			std::set<int> labels;
			for (int j = 0; j < dim; j++) {
				if (cur_bottom_label[j] > 0) labels.insert(j);
			}

			CHECK_GT(labels.size(), 0);
			int acc = 0;
			for (int j = dim - 1; j >= dim - (int)labels.size(); j--) {
				if (labels.count(score[j].second)) acc++;
			}

			*accuracy += (data_type)acc / labels.size();
		}

		int count = prob->count();
		*loss = 0;

		// multi-label loss
		for (int i = 0; i < count; ++i) {
			double x = -log(1.0 / bottom_data[i] - 1);
			*loss -= (data_type)(x * (bottom_label[i] - (x >= 0)) - log(1 + exp(x - 2 * x * (x >= 0))));
		}

		*accuracy /= num;
		*loss /= num;
	}
}