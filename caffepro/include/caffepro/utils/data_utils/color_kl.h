
#pragma once

#include <caffepro/utils/data_utils/bigfile.h>

namespace caffepro {
	namespace data_utils{

		struct kl_info {
			float P[3][3];
			float SqrtV[3];
		};

		struct kl_infos {
			std::map<std::string, int> label2classID;
			std::map<int, std::string> classID2label;
			std::vector<std::map<std::string, kl_info> > name2ColorKL;
		};

		// color kl related functions
		void load_color_kl(__out kl_infos &kls, __in const std::string &folderPath);
		bool random_color_shift(__out float shift[3], __in const kl_infos &kls, const raw_picture &img);
		void random_color_shift(__out float shift[3], __in const kl_info &kl);
	}
}