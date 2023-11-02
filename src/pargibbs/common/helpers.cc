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

  // Compute length of final vector by summing all nnz values to check that were
  // not trying to gather too many values
  auto length = std::accumulate(nnzs.begin(), nnzs.end(), 0);
  if (mpi_helper::is_debug_rank() and length != vec.size())
    throw std::runtime_error(
        "In mpi_gather_vector: Sum of nonzeros of all vectors does not match "
        "the vector size. Make sure to remove halo values before calling this "
        "function.");

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

Eigen::MatrixXd
pargibbs::mpi_gather_matrix(const Eigen::SparseMatrix<double> &mat) {
  if (not mat.isCompressed())
    throw std::runtime_error(
        "In mpi_gather_matrix: Matrix must be compressed.");

  // Gather the outer index ptrs of mat
  std::vector<int> outer_index_ptrs;
  if (mpi_helper::is_debug_rank())
    outer_index_ptrs.resize(mpi_helper::get_size() * mat.outerSize());

  MPI_Gather(mat.outerIndexPtr(),
             mat.outerSize(),
             MPI_INT,
             outer_index_ptrs.data(),
             mat.outerSize(),
             MPI_INT,
             mpi_helper::debug_rank(),
             MPI_COMM_WORLD);

  // Gather number of nonzeros in mat (length of inner_index and value arrays)
  std::vector<int> nnzs;
  if (mpi_helper::is_debug_rank())
    nnzs.resize(mpi_helper::get_size());

  int own_nnz = mat.nonZeros();
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

  auto length = mat.rows() * mat.cols();

  // Next, gather the inner index ptrs of mat
  std::vector<int> inner_index_ptrs;
  if (mpi_helper::is_debug_rank())
    inner_index_ptrs.resize(length);

  MPI_Gatherv(mat.innerIndexPtr(),
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

  MPI_Gatherv(mat.valuePtr(),
              own_nnz,
              MPI_DOUBLE,
              value_ptrs.data(),
              nnzs.data(),
              displs.data(),
              MPI_DOUBLE,
              mpi_helper::debug_rank(),
              MPI_COMM_WORLD);

  // Finally, collect all values into one dense matrix
  Eigen::MatrixXd res;
  if (mpi_helper::is_debug_rank()) {
    res.resize(mat.rows(), mat.cols());
    res.setZero();

    for (int i = 0; i < mpi_helper::get_size(); ++i) {
      // Loop over rows and compute outer index ptr for rank i
      for (int row = 0; row < mat.outerSize(); ++row) {
        auto op_begin = outer_index_ptrs[i * mat.outerSize() + row];
        auto op_end = 0;
        // For the last row, op_end is one past the length of inner_index_ptr,
        // i.e., the number of nonzeros
        if (row == mat.outerSize() - 1)
          op_end = nnzs[i];
        else
          op_end = outer_index_ptrs[i * mat.outerSize() + row + 1];

        for (int j = op_begin; j < op_end; ++j) {
          int col = inner_index_ptrs[displs[i] + j];
          res.coeffRef(row, col) = value_ptrs[displs[i] + j];
        }
      }
    }
  }

  return res;
}
