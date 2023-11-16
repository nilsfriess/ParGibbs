#pragma once

#include "pargibbs/lattice/lattice.hh"

#include <cstddef>
#include <cassert>
#include <memory>

namespace pargibbs {
template <class Mat, class Vec> class LatticeOperator {
public:
  using Matrix = Mat;
  using Vector = Vec;

  template <class MatBuilder>
  LatticeOperator(std::size_t dim, Lattice::IndexType lattice_size,
                  MatBuilder &&matrix_builder)
      : lattice_{dim, lattice_size}, matrix_{matrix_builder(lattice_)},
        vector_(matrix_.rows()) {
    vector_.setZero();
  }

  const Lattice &get_lattice() const { return lattice_; }

  const Matrix &get_matrix() const { return matrix_; }

  Vector &vector() { return vector_; }

  std::size_t size() const { return lattice_.get_n_total_vertices(); };

  template <class CoarseMatBuilder>
  std::shared_ptr<LatticeOperator>
  coarsen(CoarseMatBuilder &&coarse_matrix_builder) const {
    auto coarse_lattice = lattice_.coarsen();
    auto coarse_matrix = coarse_matrix_builder(coarse_lattice, matrix_);
    assert(coarse_matrix.rows() == coarse_lattice.get_n_total_vertices() &&
           coarse_matrix.cols() == coarse_lattice.get_n_total_vertices());

    // Can't use make_shared since the constructor is private
    return std::shared_ptr<LatticeOperator>(new LatticeOperator(
        std::move(coarse_lattice), std::move(coarse_matrix)));
  }

private:
  LatticeOperator(Lattice &&lattice, Matrix &&matrix)
      : lattice_{std::move(lattice)}, matrix_{std::move(matrix)},
        vector_(matrix_.rows()) {
    vector_.setZero();
  };

  Lattice lattice_;
  Matrix matrix_;
  Vector vector_;
};

} // namespace pargibbs
