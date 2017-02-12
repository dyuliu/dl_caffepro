
#include <caffepro\utils\multinode.h>

namespace caffepro {

	caffepro::multinode* caffepro::multinode::instance_ = new caffepro::multinode();

	void multinode::init(int argc, char *argv[]) {
		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &worker_id_);
		MPI_Comm_size(MPI_COMM_WORLD, &worker_size_);
		worker_id_show_ = int_to_string(worker_id_);
		COUT_WORKER(worker_id_show_, "Prepared to work together") << std::endl;
	}

	void multinode::all_sum(data_type *data, int size) {
		MPI_Allreduce(MPI_IN_PLACE, (void*)data, (int)size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	}
}