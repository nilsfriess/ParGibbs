#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <iostream>
#include <pargibbs/forward_substitution.hh>

TEST(SolverTest, ForwardSubstitution) {
  using Matrix = Eigen::Matrix<double, 4, 4>;
  using Vector = Eigen::Vector<double, 4>;

  pargibbs::ForwardSubstitutionSolver<Matrix, Vector> solver;

  Matrix matrix{{3, 0, 0, 0}, {-1, 1, 0, 0}, {3, -2, -1, 0}, {1, -2, 6, 2}};
  Vector rhs{5, 6, 4, 2};

  Vector solution = solver.solve(matrix, rhs);
  Vector reference = matrix.fullPivLu().solve(rhs);

  EXPECT_NEAR((solution - reference).norm(), 0, 1e-12);
}
