#pragma once

#include <caffepro\caffepro.h>
#include <caffepro\utils\string_uitls.h>
#include <caffepro\utils\color_print.h>

#define MPICH_SKIP_MPICXX
#include "mpi.h"

namespace caffepro {

	class multinode {

	private:
		multinode() {
		}

		multinode(const multinode &ptr);
		multinode operator=(const multinode &ptr);
		
	private:
		static multinode* instance_;
		int worker_id_;
		int worker_size_;
		std::string worker_id_show_;

	public:
		static multinode* get() {
			if (instance_ == NULL) instance_ = new multinode();
			return instance_;
		}
		
		~multinode() {
			MPI_Finalize();
			delete instance_;
		}

		// init
		void init(int argc, char *argv[]);

		// function
		void all_sum(data_type *data, int size);
		 
		double now() {
			return MPI_Wtime();
		}

		void barrier() {
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// data
		std::string get_worker_id_show() { return worker_id_show_; }
		int get_worker_id() { return worker_id_; }
		int get_worker_size() { return worker_size_; }
	};
}