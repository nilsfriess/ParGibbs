#include "pargibbs/common/helpers.hh"
#include "pargibbs/mpi_helper.hh"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <numeric>
#include <vector>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

Eigen::VectorXd
pargibbs::mpi_gather_vector(const Eigen::SparseVector<double> &vec) {
  // Gather number of nonzeros in vec
  std::vector<int> nnzs;
  if (mpi_helper::is_debug_rank())
    nnzs.resize(mpi_helper::get_size());

  int own_nnz = vec.nonZeros();
  MPI_Gather(&own_nnz,
             1,
             MPI_INT,
             nnzs.data(),
             1,
             MPI_INT,
             mpi_helper::debug_rank(),
             MPI_COMM_WORLD);

  std::vector<int> displs;
  if (mpi_helper::is_debug_rank()) {
    displs.resize(mpi_helper::get_size());
    displs[0] = 0;
    for (std::size_t i = 1; i < displs.size(); ++i)
      displs[i] = displs[i - 1] + nnzs[i - 1];
  }

  // Compute length of final vector by summing all nnz values
  auto length = std::accumulate(nnzs.begin(), nnzs.end(), 0);

  // Next, gather the inner index ptrs of the sparse vectors
  std::vector<int> inner_index_ptrs;
  if (mpi_helper::is_debug_rank())
    inner_index_ptrs.resize(length);

  MPI_Gatherv(vec.innerIndexPtr(),
              own_nnz,
              MPI_INT,
              inner_index_ptrs.data(),
              nnzs.data(),
              displs.data(),
              MPI_INT,
              mpi_helper::debug_rank(),
              MPI_COMM_WORLD);

  // Next, gather the actual values
  std::vector<double> value_ptrs;
  if (mpi_helper::is_debug_rank())
    value_ptrs.resize(length);

  MPI_Gatherv(vec.valuePtr(),
              own_nnz,
              MPI_DOUBLE,
              value_ptrs.data(),
              nnzs.data(),
              displs.data(),
              MPI_DOUBLE,
              mpi_helper::debug_rank(),
              MPI_COMM_WORLD);

  // Finally, collect all values into one vector
  Eigen::VectorXd res(length);
  if (mpi_helper::is_debug_rank())
    for (int i = 0; i < length; ++i)
      res[inner_index_ptrs[i]] = value_ptrs[i];

  return res;
}
