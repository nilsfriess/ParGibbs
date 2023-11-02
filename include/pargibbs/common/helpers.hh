#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace pargibbs {
// Gathers Eigen::SparseVectors that are scattered across multiple MPI processes
// into one dense Eigen::VectorXd on the rank that is returned by
// mpi_helper::debug_rank(). Assumes that the vectors are non-overlapping, i.e.,
// potential halo values have to be removed first.
// This is mostly for debugging purposes as it calls MPI_Gather[v] multiple
// times.
Eigen::VectorXd mpi_gather_vector(const Eigen::SparseVector<double> &vec);

// Same as mpi_gather_vector but for matrices. This will compress the matrix if
// it is not already compressed.
Eigen::MatrixXd mpi_gather_matrix(const Eigen::SparseMatrix<double> &mat);
} // namespace pargibbs
