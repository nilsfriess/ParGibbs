#pragma once

#include <pargibbs/pargibbs.hh>

#include <Eigen/SparseCore>
#include <vector>

// This operator is represented by a matrix that comes from Example 6.1 from
// [Fox, Parker: Accelerated Gibbs sampling of normal distributions using matrix
// splittings and polynomials, https://arxiv.org/abs/1505.03512]

template <class Lattice> class GMRFOperator {
  using Triplet = Eigen::Triplet<double>;
  using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

public:
  using MatrixType = SparseMatrix;

  GMRFOperator(const Lattice &lattice) : lattice(lattice) {
    constexpr int nnz = 460;
    std::vector<Triplet> triplets;
    triplets.reserve(nnz);

    const double noise_var = 1e-4;

    for (auto v : lattice.own_vertices) {
      int n_neighbours = lattice.adj_idx[v + 1] - lattice.adj_idx[v];
      triplets.emplace_back(v, v, n_neighbours + noise_var);

      for (typename Lattice::IndexT n = lattice.adj_idx[v];
           n < lattice.adj_idx[v + 1]; ++n) {
        auto nb_idx = lattice.adj_vert[n];
        triplets.emplace_back(v, nb_idx, -1);
      }
    }

    auto mat_size = lattice.get_n_total_vertices();
    prec = SparseMatrix(mat_size, mat_size);
    prec.setFromTriplets(triplets.begin(), triplets.end());
  }

  const SparseMatrix &get_matrix() const { return prec; }
  const Lattice &get_lattice() const { return lattice; }

private:
  const Lattice &lattice;

  SparseMatrix prec;
};
