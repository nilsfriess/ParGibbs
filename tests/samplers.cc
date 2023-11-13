#include "pargibbs/common/helpers.hh"
#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/lattice/types.hh"
#include "pargibbs/samplers/gibbs.hh"
#include "pargibbs/samplers/multigrid.hh"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>

std::pair<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::MatrixXd>
get_test_matrices(const pargibbs::Lattice &lattice, bool compute_cov = false) {
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

  if (compute_cov) {
    Eigen::MatrixXd dense_precision = precision;
    auto covariance = dense_precision.inverse();
    return {precision, covariance};
  } else
    return {precision, Eigen::MatrixXd{}};
}

TEST(SamplersTest, Gibbs1D) {
  namespace pg = pargibbs;

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  pg::Lattice lattice(1, 8);

  auto [precision, covariance] = get_test_matrices(lattice, true);

  pg::GibbsSampler sampler(
      std::make_shared<pg::Lattice>(lattice),
      std::make_shared<Eigen::SparseMatrix<double, Eigen::RowMajor>>(precision),
      &engine,
      1.68);
  sampler.enable_estimate_covariance();

  const std::size_t n_burnin = 1000;
  const std::size_t n_samples = 1'000'000;

  Eigen::VectorXd sample(lattice.get_n_total_vertices());
  sample.setZero();

  Eigen::VectorXd nu_mean(lattice.get_n_total_vertices());
  std::uniform_real_distribution<double> real_dist(0, 1);
  pargibbs::for_each_ownindex_and_halo(
      lattice, [&](auto idx) { nu_mean.coeffRef(idx) = real_dist(engine); });

  sampler.sample(sample, nu_mean, n_burnin);
  sampler.reset_statistics();

  sampler.sample(sample, nu_mean, n_samples);

  const double tol = 5e-3;
  // Expect relative error for sample covariance matrix to be near zero
  EXPECT_NEAR(1. / covariance.norm() *
                  (sampler.get_covariance() - covariance).norm(),
              0,
              tol);
}

TEST(SamplersTest, Gibbs1DRedBlack) {
  namespace pg = pargibbs;

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  pg::Lattice lattice(
      1, 8, pg::ParallelLayout::None, pg::LatticeOrdering::RedBlack);

  auto [precision, covariance] = get_test_matrices(lattice, true);

  pg::GibbsSampler sampler(
      std::make_shared<pg::Lattice>(lattice),
      std::make_shared<Eigen::SparseMatrix<double, Eigen::RowMajor>>(precision),
      &engine,
      1.68);
  sampler.enable_estimate_mean();
  sampler.enable_estimate_covariance();

  const std::size_t n_burnin = 1000;
  const std::size_t n_samples = 1'000'000;

  Eigen::VectorXd sample(lattice.get_n_total_vertices());
  sample.setZero();

  Eigen::VectorXd nu_mean(lattice.get_n_total_vertices());
  std::uniform_real_distribution<double> real_dist(0, 1);
  pargibbs::for_each_ownindex_and_halo(
      lattice, [&](auto idx) { nu_mean.coeffRef(idx) = real_dist(engine); });

  sampler.sample(sample, nu_mean, n_burnin);
  sampler.reset_statistics();

  sampler.sample(sample, nu_mean, n_samples);

  const double tol = 5e-3;
  // Expect relative error for sample covariance matrix to be near zero
  EXPECT_NEAR(1. / covariance.norm() *
                  (sampler.get_covariance() - covariance).norm(),
              0,
              tol);
}

TEST(SamplersTest, Multigrid2d) {
  namespace pg = pargibbs;

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  pg::Lattice lattice(2, 9);
  std::cout << "Generating test matrices..." << std::flush;
  auto [precision, covariance] = get_test_matrices(lattice, true);
  std::cout << " done." << std::endl;

  using Sampler = pg::
      MultigridSampler<Eigen::VectorXd, decltype(precision), decltype(engine)>;

  Sampler::Parameters params{
      .levels = 3, .cycles = 2, .n_presample = 2, .n_postsample = 2};

  Sampler sampler(
      std::make_shared<pg::Lattice>(lattice),
      std::make_shared<Eigen::SparseMatrix<double, Eigen::RowMajor>>(precision),
      &engine,
      params);
  sampler.enable_estimate_mean();
  sampler.enable_estimate_covariance();

  const std::size_t n_burnin = 1000;
  const std::size_t n_samples = 1'000'000;

  Eigen::VectorXd sample(lattice.get_n_total_vertices());
  sample.setZero();

  Eigen::VectorXd nu_mean(lattice.get_n_total_vertices());
  nu_mean.setZero();

  sampler.sample(sample, nu_mean, n_burnin);
  sampler.reset_statistics();

  sampler.sample(sample, nu_mean, n_samples);

  const double tol = 8e-3;
  // Expect relative error for sample covariance matrix to be near zero
  EXPECT_NEAR(1. / covariance.norm() *
                  (sampler.get_covariance() - covariance).norm(),
              0,
              tol);
}
