#pragma once

#include "pargibbs/lattice/lattice.hh"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <set>

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

template <class Functor>
void for_each_ownindex_and_halo(const Lattice &lattice, Functor f) {
  std::set<typename Lattice::IndexType> indices;
  for (auto v : lattice.own_vertices) {
    indices.insert(v);
    for (int n = lattice.adj_idx.at(v); n < lattice.adj_idx.at(v + 1); ++n) {
      auto nb_idx = lattice.adj_vert.at(n);
      indices.insert(nb_idx);
    }
  }

  for (auto index : indices)
    f(index);
}
} // namespace pargibbs
