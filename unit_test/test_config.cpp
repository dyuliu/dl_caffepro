
#include <caffepro/object_model/caffepro_config.h>
#include <iostream>

using namespace caffepro;
using namespace std;

void test_config() {
	caffepro_config config;
	config.load_config("net.cfg");

	caffepro_config_reader reader(config, "fc");
	caffepro_config_reader reader2(config, "");
	caffepro_config_reader reader3(config, "fc2");

	string dff1 = reader.get<string>("dff");
	int dff2 = reader2.get<int>("dff");
	int dff3 = reader3.get<int>("dff");
	vector<float> ko = reader3.get_array<float>("ko");
	vector<string> aaaa = reader2.get_array<string>("fff");
	vector<string> aaaaa = reader2.get_array<string>("ffff", false);

	int ds = reader2.get<int>("ds", true, 333);
}

//int main() {
//	test_config();
//}