#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/samplers/gibbs.hh"

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>

std::pair<Eigen::SparseMatrix<double>, Eigen::MatrixXd>
get_test_matrices(const pargibbs::Lattice &lattice) {
  const auto size = lattice.get_n_total_vertices();

  Eigen::SparseMatrix<double> precision(size, size);
  std::vector<Eigen::Triplet<double>> triplets;

  const double diag = 100;
  const double off_diag = 0.01;

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

TEST(SamplersTest, Gibbs_ParallelLayoutNone) {
  namespace pg = pargibbs;

  std::mt19937 engine;

  pg::Lattice lattice(2, 5);

  auto [precision, covariance] = get_test_matrices(lattice);

  pg::GibbsSampler sampler(&lattice, &precision, &engine, true, 1.98);

  const std::size_t n_burnin = 1000;
  const std::size_t n_samples = 500000;

  Eigen::SparseVector<double> sample(lattice.get_n_total_vertices());

  sampler.sample(sample, n_burnin);
  sampler.reset_mean();

  for (std::size_t n = 0; n < n_samples; ++n) {
    sampler.sample(sample, 1);
  }

  EXPECT_NEAR(sampler.get_mean().norm(), 0, 1e-4);
}
