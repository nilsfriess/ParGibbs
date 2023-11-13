#include "pargibbs/common/helpers.hh"
#include "pargibbs/lattice/helpers.hh"
#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/mpi_helper.hh"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <numeric>
#include <stdexcept>
#include <vector>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

Eigen::VectorXd pargibbs::mpi_gather_vector(const Eigen::VectorXd &vec,
                                            const Lattice &lattice) {
  // Gather number of vertices
  std::vector<int> n_vertices;
  if (mpi_helper::is_debug_rank())
    n_vertices.resize(mpi_helper::get_size());

  auto own_n_vertices = lattice.own_vertices.size();
  MPI_Gather(&own_n_vertices,
             1,
             MPI_INT,
             n_vertices.data(),
             1,
             MPI_INT,
             mpi_helper::debug_rank(),
             MPI_COMM_WORLD);

  std::vector<int> displs;
  if (mpi_helper::is_debug_rank()) {
    displs.resize(mpi_helper::get_size());
    displs[0] = 0;
    for (std::size_t i = 1; i < displs.size(); ++i)
      displs[i] = displs[i - 1] + n_vertices[i - 1];
  }

  std::vector<int> indices;
  if (mpi_helper::is_debug_rank())
    indices.resize(vec.size());

  MPI_Gatherv(lattice.own_vertices.data(),
              own_n_vertices,
              MPI_INT,
              indices.data(),
              n_vertices.data(),
              displs.data(),
              MPI_INT,
              mpi_helper::debug_rank(),
              MPI_COMM_WORLD);

  std::vector<double> own_values(own_n_vertices);
  for (std::size_t i = 0; i < own_n_vertices; ++i)
    own_values[i] = vec[lattice.own_vertices[i]];

  std::vector<double> values;
  if (mpi_helper::is_debug_rank())
    values.resize(vec.size());

  MPI_Gatherv(own_values.data(),
              own_n_vertices,
              MPI_DOUBLE,
              values.data(),
              n_vertices.data(),
              displs.data(),
              MPI_DOUBLE,
              mpi_helper::debug_rank(),
              MPI_COMM_WORLD);

  Eigen::VectorXd res(values.size());
  if (mpi_helper::is_debug_rank())
    for (std::size_t i = 0; i < values.size(); ++i)
      res[indices[i]] = values[i];

  return res;
}

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

Eigen::SparseMatrix<double>
pargibbs::make_prolongation(const pargibbs::Lattice &fine,
                            const pargibbs::Lattice &coarse) {
  if (fine.dim != 2 or coarse.dim != 2)
    throw std::runtime_error(
        "make_prolongation: Only dim = 2 supported currently.");

  if (fine.dim != coarse.dim)
    throw std::runtime_error("make_prolongation: Cannot create prolongation "
                             "operator for lattices of different dimension.");

  if (fine.get_vertices_per_dim() < coarse.get_vertices_per_dim())
    throw std::runtime_error(
        "make_prolongation: Fine lattice must be finer than coarse lattice.");

  std::vector<Eigen::Triplet<double>> triplets;

  // We construct the matrix in "block" that correspond to rows of the fine
  // lattice. There are two types of blocks: Those that correspond to rows that
  // share some points with the coarse grid and those that do not exist in the
  // coarse grid.

  using Idx = Lattice::IndexType;
  Idx start = 0;
  for (Idx block = 0; block < fine.get_vertices_per_dim(); ++block) {
    for (Idx block_row = 0; block_row < fine.get_vertices_per_dim();
         ++block_row) {
      const auto row = fine.get_vertices_per_dim() * block + block_row;

      // First type of "block"
      if (block % 2 == 0) {
        if (block_row % 2 == 0) {
          triplets.emplace_back(row, start + block_row / 2, 1);
        } else {
          triplets.emplace_back(row, start + (block_row - 1) / 2, 0.5);
          triplets.emplace_back(row, start + 1 + (block_row - 1) / 2, 0.5);
        }

      } else { // Second type of "block"
        if (block_row % 2 == 0) {
          triplets.emplace_back(row, start + block_row / 2, 0.5);
          triplets.emplace_back(
              row, start + coarse.get_vertices_per_dim() + block_row / 2, 0.5);
        } else {
          triplets.emplace_back(row, start + (block_row - 1) / 2, 0.25);
          triplets.emplace_back(row, start + 1 + (block_row - 1) / 2, 0.25);
          triplets.emplace_back(row,
                                start + coarse.get_vertices_per_dim() +
                                    (block_row - 1) / 2,
                                0.25);
          triplets.emplace_back(row,
                                start + coarse.get_vertices_per_dim() + 1 +
                                    (block_row - 1) / 2,
                                0.25);
        }
      }
    }

    if (block % 2 == 1)
      start += coarse.get_vertices_per_dim();
  }

  Eigen::SparseMatrix<double> mat(fine.get_n_total_vertices(),
                                  coarse.get_n_total_vertices());
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}

Eigen::SparseMatrix<double>
pargibbs::make_restriction(const pargibbs::Lattice &fine,
                           const pargibbs::Lattice &coarse) {
  return make_prolongation(fine, coarse).transpose();
}
