#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "pargibbs/lattice/lattice.hh"

inline Eigen::SparseMatrix<double>
gmrf_matrix_builder(const pargibbs::Lattice &lattice) {
  const int entries_per_row = 5;
  const int nnz = lattice.own_vertices.size() * entries_per_row;
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nnz);

  const double noise_var = 1e-4;

  auto handle_row = [&](auto v) {
    int n_neighbours = lattice.adj_idx[v + 1] - lattice.adj_idx[v];
    triplets.emplace_back(v, v, n_neighbours + noise_var);

    for (typename pargibbs::Lattice::IndexType n = lattice.adj_idx[v];
         n < lattice.adj_idx[v + 1];
         ++n) {
      auto nb_idx = lattice.adj_vert[n];
      triplets.emplace_back(v, nb_idx, -1);
    }
  };

  for (auto v : lattice.own_vertices)
    handle_row(v);

  auto mat_size = lattice.get_n_total_vertices();
  Eigen::SparseMatrix<double> matrix(mat_size, mat_size);
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  return matrix;
}
