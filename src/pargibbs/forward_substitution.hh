#pragma once

namespace pargibbs {
template <class Matrix, class Vector> struct ForwardSubstitutionSolver {
  Vector solve(const Matrix &matrix, const Vector &rhs) {
    const auto size = matrix.rows();

    Vector res;
    res[0] = rhs[0] / matrix(0, 0);
    for (auto i = 1; i < size; ++i) {
      res[i] = rhs[i];
      for (auto j = 0; j < i; ++j)
        res[i] -= matrix(i, j) * res[j];
      res[i] /= matrix(i, i);
    }

    return res;
  }
};
}; // namespace pargibbs
