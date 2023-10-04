#pragma once

namespace pargibbs {
template <class Matrix> struct MatrixOperator {
  using MatrixType = Matrix;
  
  explicit MatrixOperator(const Matrix &matrix) : matrix(matrix) {}

  const Matrix &get_matrix() const { return matrix; }

  const Matrix &matrix;
};
} // namespace pargibbs
