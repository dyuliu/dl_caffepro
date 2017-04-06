#include <iostream>
#include <string>

namespace analyzer_proto {
	class Info;
	class Recorders;
	class Images;
}

namespace analyzer_tools {

	namespace analyzer {
		class Infos;
		class Recorders;
	}

	namespace db {
		class DB;
	}

	using std::string;
	using analyzer::Infos;
	using analyzer::Recorders;
	using analyzer_proto::Info;
	using analyzer_proto::Images;
	
	void print();

	bool save_image(char *data, int length, string dir_name, string file_name);

	class Analyzer {
	public:
		enum class RECORD_TYPE : unsigned int {
			TRAIN_ERROR = 0U,
			TRAIN_LOSS = 1U,
			TEST_ERROR = 2U,
			TEST_LOSS = 3U,
			FORWARD_TIME = 4U,
			BACKWARD_TIME = 5U,
			UPDATE_TIME = 6U,
			LEARNING_RATE = 7U
		};


	public:
		Analyzer(string db_name, string model_name, string host = "localhost:27017");
		~Analyzer();

		bool deal_rec_info(int iteration, RECORD_TYPE type, float value);

		bool deal_para_info(Info &para_info_);

		bool deal_img_info(Images &img_info_, int batchsize);

	private:
		db::DB *dbInstance;
		bool firstParaInfo, firstImgInfo;
		std::shared_ptr<Infos> pre_info;
		std::map<std::string, int> map_img_label;
		std::map<RECORD_TYPE, std::string> recType;
	};
	
}

