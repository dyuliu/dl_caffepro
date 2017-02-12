
#include <caffepro/object_model/caffepro_net.h>
#include <caffepro/proto/proto_io.h>
#include <caffepro/context/common_names.h>
#include <caffepro/layers/data_bigfile_layer.h>
#include <iostream>

using namespace caffepro;
using namespace std;

void run() {
	// model file names
	const string prototxt_file = "net_test.prototxt";
	const string model_file = "net_final_bnst.model";

	// load model
	NetParameter net_param;
	proto_io(net_param).from_text_file(prototxt_file);

	caffepro_context context;
	caffepro_net *net = caffepro_net::create_from_proto(&context, net_param);
	
	NetParameter net_weights;
	proto_io(net_weights).from_binary_file(model_file);
	net->load_weights(net_weights);

	// specify where to get scores
	const string score_entry_name = "fc10"; // usually, it is the first bottom name of loss layer
	device_blob &score_source = *net->get_blob(score_entry_name)->get(0);

	// start running
	// for each iteration, we will process a batch
	int num_of_batches = 200;
	for (int batch = 0; batch < num_of_batches; batch++) {
		// forward
		net->forward();
		context.sync_all_devices(); // wait for forward finished

		// after forward, images of the current batch are processed. We only need to extract the results we need

		// get picture info in the batch
		data_bigfile_layer *data_layer = dynamic_cast<data_bigfile_layer *>(context.get_shared_object(net->get_name(), SHAREDOBJNAME_DATASOURCE));
		CHECK(data_layer);

		auto &images = data_layer->current_batch().processing_imgs; // images in the batch
		int batch_size = (int)images.size(); // number of images, i.e. batch size

		CHECK_EQ(batch_size, score_source.num());

		// then we process each image
		for (int im = 0; im < batch_size; im++) {
			// get image score vector
			int score_dim = score_source.inner_count();
			const data_type *score_vec = score_source.cpu_data() + score_source.offset(im);

			// here, score_dim = number of classes, and score_vec[i] means the score of the probability belonging to class[i] (i is zero based)
			// Note that the score is NOT normalized by sigmoid or softmax. You must do it manually if you want to use the score for combination

			// get metadata for this image
			string picture_name = images[im].path;
			int label_id = images[im].label_id; // ground-truth label for the image. The item is INVALID when using multi-label model

			// an example as follows: output the top 5 classes for the image
			{
				vector<pair<data_type, int> > rank;
				for (int i = 0; i < score_dim; i++) {
					rank.push_back(make_pair(score_vec[i], i));
				}
				sort(rank.begin(), rank.end());

				cout << "Picture Name: " << picture_name << endl;
				cout << "GroundTruth Label: " << label_id << endl;
				for (int i = score_dim - 1; i >= score_dim - 5; i--) {
					cout << "Top " << score_dim - i << ": " << rank[i].second << " (Conf: " << rank[i].first << ")" << endl;
					// conf here is not normalized
				}
				cout << endl;
				system("pause");
			}
		}
	}

	// delete net
	delete net;
}

//int main() {
//	run();
//}