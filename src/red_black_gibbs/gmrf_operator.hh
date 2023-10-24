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
    std::vector<Triplet> triplets(nnz);

    const double noise_var = 1e-4;

    const auto rank = pargibbs::mpi_helper::get_rank();
    for (std::size_t v = 0; v < lattice.get_n_total_vertices(); ++v) {
      if (lattice.mpiowner[v] != rank)
        continue;

      int n_neighbours = 0;
      for (int n = 1; n < 5; ++n) {
        if (lattice.vertices[5 * v + n] != -1) {
          n_neighbours++;
          triplets.emplace_back(v, lattice.vertices[5 * v + n], -1);
        }
      }

      triplets.emplace_back(v, v, n_neighbours + noise_var);
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
