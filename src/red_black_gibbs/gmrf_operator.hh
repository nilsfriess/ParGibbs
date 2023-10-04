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

    const auto add_triplet_for_point =
        [&](const pargibbs::LatticePoint<Lattice::Dim> &lattice_point) {
          const auto row = lattice_point.actual_index;
          triplets.emplace_back(row, row, 2);

          for (const auto neighbour : lattice.get_neighbours(lattice_point)) {
            const auto col = neighbour.actual_index;
            triplets.emplace_back(row, col, -1);
          }
        };

    const auto red_points = lattice.get_my_points().first;
    const auto black_points = lattice.get_my_points().second;

    // Handle red points (or all points in case of sequential execution)
    std::for_each(red_points.begin(), red_points.end(), add_triplet_for_point);

    // Handle black points (or no points in case of sequential execution)
    std::for_each(black_points.begin(), black_points.end(),
                  add_triplet_for_point);

    prec = SparseMatrix(lattice.get_total_points(), lattice.get_total_points());
    prec.setFromTriplets(triplets.begin(), triplets.end());
  }

  const SparseMatrix &get_matrix() const { return prec; }
  const Lattice &get_lattice() const { return lattice; }

private:
  const Lattice &lattice;

  SparseMatrix prec;
};
