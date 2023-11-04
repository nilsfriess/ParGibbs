#include "pargibbs/common/helpers.hh"
#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/samplers/gibbs.hh"

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>

std::pair<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::MatrixXd>
get_test_matrices(const pargibbs::Lattice &lattice) {
  const auto size = (std::size_t)lattice.get_n_total_vertices();

  Eigen::SparseMatrix<double, Eigen::RowMajor> precision(size, size);
  std::vector<Eigen::Triplet<double>> triplets;

  const double diag = 6;
  const double off_diag = -1;

  for (std::size_t i = 0; i < size; ++i) {
    if (i > 0)
      triplets.emplace_back(i, i - 1, off_diag);

    triplets.emplace_back(i, i, diag);

    if (i < size - 1)
      triplets.emplace_back(i, i + 1, off_diag);
  }
  precision.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::MatrixXd dense_precision = precision;
  auto covariance = dense_precision.inverse();

  return {precision, covariance};
}

TEST(SamplersTest, Gibbs1D) {
  namespace pg = pargibbs;

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  pg::Lattice lattice(1, 8);

  auto [precision, covariance] = get_test_matrices(lattice);

  pg::GibbsSampler sampler(&lattice, &precision, &engine, 1.68);
  sampler.enable_estimate_mean();
  sampler.enable_estimate_covariance();

  const std::size_t n_burnin = 100;
  const std::size_t n_samples = 1'000'000;

  Eigen::SparseVector<double> sample(lattice.get_n_total_vertices());
  pargibbs::for_each_ownindex_and_halo(
      lattice, [&](auto idx) { sample.insert(idx) = 0; });

  sampler.sample(sample, n_burnin);
  sampler.reset_statistics();

  sampler.sample(sample, n_samples);

  const double tol = 5e-3;
  // Expect mean to be near zero
  EXPECT_NEAR(sampler.get_mean().norm(), 0, tol);
  // Expect relative error for sample covariance matrix to be near zero
  EXPECT_NEAR(1. / covariance.norm() *
                  (sampler.get_covariance() - covariance).norm(),
              0,
              tol);
}
