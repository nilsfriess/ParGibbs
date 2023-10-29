#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "pargibbs/lattice/lattice.hh"

// This operator is represented by a matrix that comes from Example 6.1 from
// [Fox, Parker: Accelerated Gibbs sampling of normal distributions using matrix
// splittings and polynomials, https://arxiv.org/abs/1505.03512]

struct GMRFOperator {
  using Triplet = Eigen::Triplet<double>;
  using SparseMatrix = Eigen::SparseMatrix<double>;

  GMRFOperator(const pargibbs::Lattice &lattice) {
    const int entries_per_row = 5;
    const int nnz = lattice.own_vertices.size() * entries_per_row;
    std::vector<Triplet> triplets;
    triplets.reserve(nnz);

    const double noise_var = 1e-4;

    for (auto v : lattice.own_vertices) {
      int n_neighbours = lattice.adj_idx[v + 1] - lattice.adj_idx[v];
      triplets.emplace_back(v, v, n_neighbours + noise_var);

      for (typename pargibbs::Lattice::IndexType n = lattice.adj_idx[v];
           n < lattice.adj_idx[v + 1]; ++n) {
        auto nb_idx = lattice.adj_vert[n];
        triplets.emplace_back(v, nb_idx, -1);
      }
    }

    auto mat_size = lattice.get_n_total_vertices();
    matrix = SparseMatrix(mat_size, mat_size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
  }

  SparseMatrix matrix;
};
