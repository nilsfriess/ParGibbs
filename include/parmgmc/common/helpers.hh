#pragma once

#include "parmgmc/lattice/lattice.hh"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <set>

namespace parmgmc {
// Gathers Eigen::VectorXds that are scattered across multiple MPI processes
// into one Eigen::VectorXd on the rank that is returned by
// mpi_helper::debug_rank(). The lattice is required since it stores the
// vertices that the respective MPI process owns.
Eigen::VectorXd mpi_gather_vector(const Eigen::VectorXd &vec,
                                  const Lattice &lattice);

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

Eigen::SparseMatrix<double> make_prolongation(const Lattice &fine,
                                              const Lattice &coarse);
Eigen::SparseMatrix<double> make_restriction(const Lattice &fine,
                                             const Lattice &coarse);

template <class Functor>
void for_each_ownindex_and_halo(const Lattice &lattice, Functor f) {
  for (auto v : lattice.vertices(VertexType::Any))
    f(v);
}
} // namespace parmgmc
