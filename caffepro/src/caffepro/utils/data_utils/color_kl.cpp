
#include <caffepro/utils/data_utils/color_kl.h>
#include <caffepro/utils/random_helper.h>

#include <fstream>

using std::map;
using std::string;
using std::ifstream;

#define KL_FLOAT 0.1

using caffepro::random_helper;

namespace caffepro {
	namespace data_utils {

		// the path must NOT be end with '/' or '\\'!
		void load_color_kl(__out kl_infos &kls, __in const string &folderPath) {
			kls.name2ColorKL.resize(kls.label2classID.size());
			for (map<string, int>::iterator iter = kls.label2classID.begin();
				iter != kls.label2classID.end(); ++iter) {
				map<string, kl_info> &mp = kls.name2ColorKL[iter->second];
				string path = folderPath + "/" + iter->first;

				ifstream in(path);

				string pic_name_buf;
				while (in >> pic_name_buf) {
					kl_info kl;
					for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
							in >> kl.P[j][i]; // col first, so P[j][i]
						}
					}
					for (int i = 0; i < 3; i++) {
						in >> kl.SqrtV[i];

						if (kl.SqrtV[i] > 0) {
							kl.SqrtV[i] = sqrt(kl.SqrtV[i]);
						}
						else {
							kl.SqrtV[i] = 0;
						}
					}

					// skip the last 3 numbers
					// which stand for means
					float tmp;
					for (int i = 0; i < 3; i++) {
						in >> tmp;
					}

					mp[string(pic_name_buf)] = kl;
				}
			}
		}

		bool random_color_shift(__out float shift[3], __in const kl_infos &kls, const raw_picture &img) {
			shift[0] = 0; 
			shift[1] = 0; 
			shift[2] = 0;

			// modify picture pixels respected to color KL matrix
			if (kls.name2ColorKL.size() > img.label_id &&
				kls.name2ColorKL[img.label_id].count(img.path) > 0) {
				const kl_info &kl = kls.name2ColorKL[img.label_id].find(img.path)->second;

				float a[3];
				a[0] = (float)(random_helper::normal_real() * KL_FLOAT);
				a[1] = (float)(random_helper::normal_real() * KL_FLOAT);
				a[2] = (float)(random_helper::normal_real() * KL_FLOAT);

				for (int k = 0; k < 3; k++) {
					for (int j = 0; j < 3; j++) {
						shift[k] += kl.P[k][j] * kl.SqrtV[j] * a[j];
					}
				}
				return true;
			}
			return false;
		}

		void random_color_shift(__out float shift[3], __in const kl_info &kl) {
			shift[0] = 0;
			shift[1] = 0;
			shift[2] = 0;

			// modify picture pixels respected to color KL matrix
			float a[3];
			a[0] = (float)(random_helper::normal_real() * KL_FLOAT);
			a[1] = (float)(random_helper::normal_real() * KL_FLOAT);
			a[2] = (float)(random_helper::normal_real() * KL_FLOAT);

			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < 3; j++) {
					shift[k] += kl.P[k][j] * kl.SqrtV[j] * a[j];
				}
			}
		}
	}
}