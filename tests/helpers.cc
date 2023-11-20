#include <gtest/gtest.h>

#include "parmgmc/common/helpers.hh"
#include "parmgmc/lattice/lattice.hh"

TEST(IntergridTest, ProlongationMatrix2d) {
  namespace pg = parmgmc;

  pg::Lattice fine(2, 5);
  pg::Lattice coarse(2, 3);

  auto prol_matr = pg::make_prolongation(fine, coarse);

  Eigen::MatrixXd expected(25, 9);
  // clang-format off
  expected <<
  1,    0,    0,    0,    0,    0,    0,    0,    0,
  0.5,  0.5,  0,    0,    0,    0,    0,    0,    0,
  0,    1,    0,    0,    0,    0,    0,    0,    0,
  0,    0.5,  0.5,  0,    0,    0,    0,    0,    0,
  0,    0,    1,    0,    0,    0,    0,    0,    0,
  0.5,  0,    0,    0.5,  0,    0,    0,    0,    0,
  0.25, 0.25, 0,    0.25, 0.25, 0,    0,    0,    0,
  0,    0.5,  0,    0,    0.5,  0,    0,    0,    0,
  0,    0.25, 0.25, 0,    0.25, 0.25, 0,    0,    0,
  0,    0,    0.5,  0,    0,    0.5,  0,    0,    0,
  0,    0,    0,    1,    0,    0,    0,    0,    0,
  0,    0,    0,    0.5,  0.5,  0,    0,    0,    0,
  0,    0,    0,    0,    1,    0,    0,    0,    0,
  0,    0,    0,    0,    0.5,  0.5,  0,    0,    0,
  0,    0,    0,    0,    0,    1,    0,    0,    0,
  0,    0,    0,    0.5,  0,    0,    0.5,  0,    0,
  0,    0,    0,    0.25, 0.25, 0,    0.25, 0.25, 0,
  0,    0,    0,    0,    0.5,  0,    0,    0.5,  0,
  0,    0,    0,    0,    0.25, 0.25, 0,    0.25, 0.25,
  0,    0,    0,    0,    0,    0.5,  0,    0,    0.5,
  0,    0,    0,    0,    0,    0,    1,    0,    0,
  0,    0,    0,    0,    0,    0,    0.5,  0.5,  0,
  0,    0,    0,    0,    0,    0,    0,    1,    0,
  0,    0,    0,    0,    0,    0,    0,    0.5,  0.5,
  0,    0,    0,    0,    0,    0,    0,    0,    1;
  // clang-format on

  EXPECT_EQ(Eigen::MatrixXd(prol_matr), expected);
}

TEST(IntergridTest, RestrictionMatrix2d) {
  namespace pg = parmgmc;

  pg::Lattice fine(2, 5);
  pg::Lattice coarse(2, 3);

  auto restr_matr = pg::make_restriction(fine, coarse);

  Eigen::MatrixXd expected_tp(25, 9);
  // clang-format off
  expected_tp <<
  1,    0,    0,    0,    0,    0,    0,    0,    0,
  0.5,  0.5,  0,    0,    0,    0,    0,    0,    0,
  0,    1,    0,    0,    0,    0,    0,    0,    0,
  0,    0.5,  0.5,  0,    0,    0,    0,    0,    0,
  0,    0,    1,    0,    0,    0,    0,    0,    0,
  0.5,  0,    0,    0.5,  0,    0,    0,    0,    0,
  0.25, 0.25, 0,    0.25, 0.25, 0,    0,    0,    0,
  0,    0.5,  0,    0,    0.5,  0,    0,    0,    0,
  0,    0.25, 0.25, 0,    0.25, 0.25, 0,    0,    0,
  0,    0,    0.5,  0,    0,    0.5,  0,    0,    0,
  0,    0,    0,    1,    0,    0,    0,    0,    0,
  0,    0,    0,    0.5,  0.5,  0,    0,    0,    0,
  0,    0,    0,    0,    1,    0,    0,    0,    0,
  0,    0,    0,    0,    0.5,  0.5,  0,    0,    0,
  0,    0,    0,    0,    0,    1,    0,    0,    0,
  0,    0,    0,    0.5,  0,    0,    0.5,  0,    0,
  0,    0,    0,    0.25, 0.25, 0,    0.25, 0.25, 0,
  0,    0,    0,    0,    0.5,  0,    0,    0.5,  0,
  0,    0,    0,    0,    0.25, 0.25, 0,    0.25, 0.25,
  0,    0,    0,    0,    0,    0.5,  0,    0,    0.5,
  0,    0,    0,    0,    0,    0,    1,    0,    0,
  0,    0,    0,    0,    0,    0,    0.5,  0.5,  0,
  0,    0,    0,    0,    0,    0,    0,    1,    0,
  0,    0,    0,    0,    0,    0,    0,    0.5,  0.5,
  0,    0,    0,    0,    0,    0,    0,    0,    1;
  // clang-format on
  auto expected = expected_tp.transpose();

  EXPECT_EQ(Eigen::MatrixXd(restr_matr), expected);
}
