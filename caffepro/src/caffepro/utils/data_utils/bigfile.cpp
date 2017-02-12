
#include <caffepro/utils/data_utils/bigfile.h>
#include <caffepro/utils/data_utils/binary_io.h>
#include <caffepro/caffepro.h>

using std::string;
using std::vector;
using std::map;

namespace caffepro {
	namespace data_utils {
		raw_picture::raw_picture() {
			// nothing to do
		}

		raw_picture::raw_picture(int nView) {
			views.resize(nView);
			for (int i = 0; i < nView; i++) views[i] = i;
			cur_view = -1;

			width = 0;
			height = 0;
			raw_width = 0;
			raw_height = 0;
			flipped = false;
		}
		
		raw_picture::raw_picture(const std::string& s_path, int label, int nView)
			: path(s_path), label_id(label), data_length(0), data(cv::Mat()), data_offset(0) {
			views.resize(nView);
			for (int i = 0; i < nView; i++) views[i] = i;
			cur_view = -1;

			width = 0;
			height = 0;
			raw_width = 0;
			raw_height = 0;
			flipped = false;
		}

		int raw_picture::get_current_view_idx() {
			cur_view = (cur_view + 1) % int(views.size());
			if (cur_view == 0) {
				std::random_shuffle(views.begin(), views.end());
			}
			return views[cur_view];
		}

		void raw_picture::load_data() {
			data = cv::Mat(1, (int)data_length, CV_8U);
			std::ifstream in(big_file_path, std::ios::binary);
			in.seekg(data_offset, std::ios::beg);
			in.read((char *)data.ptr(), data_length);
		}

		void raw_picture::clear_data() {
			data = cv::Mat();
		}
	
		size_t load_bigfile(__out std::vector<raw_picture> &container, __in int label_ids,
			const std::string& filename, int nView, bool load_data) {
			size_t totalsize = 0;

			std::ifstream in(filename, std::ios::binary);
			char sub_file_name[300];

			while (in) {
				int n_file_name;

				if (!in.read((char *)&n_file_name, sizeof(int))) break;;
				in.read(sub_file_name, n_file_name);
				sub_file_name[n_file_name] = '\0';
				int content_length;

				in.read((char *)&content_length, sizeof(int));
				container.push_back(raw_picture(sub_file_name, label_ids, nView));
				container.back().data_length = content_length;
				container.back().data_offset = in.tellg();
				container.back().big_file_path = filename;

				if (load_data) {
					container.back().data = cv::Mat(1, content_length, CV_8U);
					in.read((char *)container.back().data.ptr(), content_length);
				}
				else {
					in.seekg(content_length, std::ios::cur);
				}

				totalsize += content_length;
			}

			return totalsize;
		}

		void load_metadata(const std::string &metadata_filename, const std::string &bigfile_folder, int nView,
			__in const std::map<std::string, int> &dic_classname2id, __inout std::vector<raw_picture> &imgs) {

			bool fill_original = !imgs.empty();
			map<string, int> picname2imageidx;

			if (fill_original) {
				// build original index
				for (int i = 0; i < (int)imgs.size(); i++) {
					picname2imageidx[imgs[i].path] = i;
				}
			}

			std::ifstream in(metadata_filename, std::ios::binary);
			binary_reader reader(in);

			int picname_maxlen = reader.read<int>();
			int clsname_maxlen = reader.read<int>();

			int processed_pics = 0;

			while (true) {
				string picname = reader.read_fixedlen_string(picname_maxlen);
				if (!in) break;

				processed_pics++;

				string clsname = reader.read_fixedlen_string(clsname_maxlen);
				CHECK(dic_classname2id.count(clsname)) << "Unknown class name in metadata file: " << clsname;

				int clsid = dic_classname2id.find(clsname)->second;

				int width = reader.read<int>();
				int height = reader.read<int>();
				int raw_width = reader.read<int>();
				int raw_height = reader.read<int>();
				int bigfile_contentlen = reader.read<int>();
				long long bigfile_contentoffset = reader.read<long long>();

				int nboxes = reader.read<int>();
				vector<Box> boxes(nboxes);
				for (int i = 0; i < nboxes; i++) {
					string box_clsname = reader.read_fixedlen_string(clsname_maxlen);
					CHECK_EQ(box_clsname, clsname) << "Class name mismatch in metadata file";

					int left = reader.read<int>();
					int top = reader.read<int>();
					int right = reader.read<int>();
					int bottom = reader.read<int>();

					boxes[i] = Box(left, top, right, bottom, clsid); // since box_clsname == clsname, clsid == box_clsid
				}

				if (fill_original) {
					CHECK(picname2imageidx.count(picname)) << "Unknown picture name in metadata file: " << picname;
					raw_picture &pic = imgs[picname2imageidx[picname]];

					// check key params
					CHECK_EQ(pic.path, picname);
					CHECK_EQ(pic.label_id, clsid);
					CHECK_EQ(pic.data_length, bigfile_contentlen);
					CHECK_EQ(pic.data_offset, bigfile_contentoffset);

					// fill other params
					pic.width = width;
					pic.height = height;
					pic.raw_width = raw_width;
					pic.raw_height = raw_height;
					pic.bounding_boxes = boxes;
				}
				else { // add new
					raw_picture pic(picname, clsid, nView);

					pic.big_file_path = bigfile_folder + "\\" + clsname + ".big";
					pic.label_id = clsid;
					pic.data_length = bigfile_contentlen;
					pic.data_offset = bigfile_contentoffset;

					// fill other params
					pic.width = width;
					pic.height = height;
					pic.raw_width = raw_width;
					pic.raw_height = raw_height;
					pic.bounding_boxes = boxes;

					imgs.push_back(pic);
				}
			}

			CHECK_EQ(imgs.size(), processed_pics);
		}

		void load_prop_boxes(const string &prop_filename, __inout vector<raw_picture> &imgs) {
			map<string, int> picname2imageidx;

			// build original index
			for (int i = 0; i < (int)imgs.size(); i++) {
				picname2imageidx[imgs[i].path] = i;
			}

			std::ifstream in(prop_filename, std::ios::binary);
			binary_reader reader(in);

			while (true) {
				string picname = reader.read<string>();
				if (!in) break;

				CHECK(picname2imageidx.count(picname) > 0);

				raw_picture &meta = imgs[picname2imageidx[picname]];

				int nboxes = reader.read<int>();
				for (int i = 0; i < nboxes; i++) {
					int left = reader.read<int>();
					int top = reader.read<int>();
					int right = reader.read<int>();
					int bottom = reader.read<int>();
					int label = reader.read<int>();;
					Box box(left, top, right, bottom, label);
					meta.prop_boxes.push_back(box);

					reader.read<float>(); // skip confidence
				}
			}
		}

		void rawpicture_to_im(__out cv::Mat &im, __in const raw_picture & rawIm) {
			im = cv::imdecode(rawIm.data, CV_LOAD_IMAGE_COLOR);
		}

		void rawpicture_to_im_adapt(__out cv::Mat &im, __in const raw_picture & rawIm) {
			im = cv::imdecode(rawIm.data, CV_LOAD_IMAGE_UNCHANGED);
		}
	}
}