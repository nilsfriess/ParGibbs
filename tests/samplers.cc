#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/lattice_operator.hh"
#include "parmgmc/lattice/lattice.hh"
#include "parmgmc/lattice/types.hh"
#include "parmgmc/samplers/gibbs.hh"
#include "parmgmc/samplers/multigrid.hh"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>

std::pair<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::MatrixXd>
get_test_matrices(const parmgmc::Lattice &lattice, bool compute_cov = false) {
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
  namespace pg = parmgmc;

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Vector = Eigen::VectorXd;
  using Operator = pg::LatticeOperator<Matrix, Vector>;

  Eigen::MatrixXd covariance;

  Operator op(1, 8, [&](const auto &lattice) {
    auto [prec, cov] = get_test_matrices(lattice, true);
    covariance = std::move(cov);

    return prec;
  });

  std::uniform_real_distribution<double> real_dist(0, 1);
  pg::for_each_ownindex_and_halo(op.get_lattice(), [&](auto idx) {
    op.vector().coeffRef(idx) = real_dist(engine);
  });

  pg::GibbsSampler sampler(std::make_shared<Operator>(op), &engine, 1.68);
  sampler.enable_estimate_covariance();

  const std::size_t n_burnin = 1000;
  const std::size_t n_samples = 1'000'000;

  Eigen::VectorXd sample(op.size());
  sample.setZero();

  sampler.sample(sample, n_burnin);
  sampler.reset_statistics();

  sampler.sample(sample, n_samples);

  const double tol = 5e-3;
  // Expect relative error for sample covariance matrix to be near zero
  EXPECT_NEAR(1. / covariance.norm() *
                  (sampler.get_covariance() - covariance).norm(),
              0,
              tol);
}

// TEST(SamplersTest, Gibbs1DRedBlack) {
//   namespace pg = parmgmc;

//   const auto seed = 0xBEEFCAFE;
//   std::mt19937 engine{seed};

//   pg::Lattice lattice(
//       1, 8, pg::ParallelLayout::None, pg::LatticeOrdering::RedBlack);

//   auto [precision, covariance] = get_test_matrices(lattice, true);
//   Eigen::VectorXd nu_mean(lattice.get_n_total_vertices());
//   std::uniform_real_distribution<double> real_dist(0, 1);
//   parmgmc::for_each_ownindex_and_halo(
//       lattice, [&](auto idx) { nu_mean.coeffRef(idx) = real_dist(engine); });

//   pg::GibbsSampler sampler(
//       std::make_shared<pg::Lattice>(lattice),
//       std::make_shared<Eigen::SparseMatrix<double,
//       Eigen::RowMajor>>(precision),
//       std::make_shared<Eigen::VectorXd>(nu_mean),
//       &engine,
//       1.68);
//   sampler.enable_estimate_mean();
//   sampler.enable_estimate_covariance();

//   const std::size_t n_burnin = 1000;
//   const std::size_t n_samples = 1'000'000;

//   Eigen::VectorXd sample(lattice.get_n_total_vertices());
//   sample.setZero();

//   sampler.sample(sample, n_burnin);
//   sampler.reset_statistics();

//   sampler.sample(sample, n_samples);

//   const double tol = 5e-3;
//   // Expect relative error for sample covariance matrix to be near zero
//   EXPECT_NEAR(1. / covariance.norm() *
//                   (sampler.get_covariance() - covariance).norm(),
//               0,
//               tol);
// }

TEST(SamplersTest, Multigrid2d) {
  namespace pg = parmgmc;

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Vector = Eigen::VectorXd;
  using Operator = pg::LatticeOperator<Matrix, Vector>;

  Eigen::MatrixXd covariance;

  Operator op(2, 8, [&](const auto &lattice) {
    auto [prec, cov] = get_test_matrices(lattice, true);
    covariance = cov;

    return prec;
  });

  std::uniform_real_distribution<double> real_dist(0, 1);
  pg::for_each_ownindex_and_halo(op.get_lattice(), [&](auto idx) {
    op.vector().coeffRef(idx) = real_dist(engine);
  });

  using Sampler = pg::MultigridSampler<decltype(op), decltype(engine)>;

  Sampler::Parameters params;
  params.levels = 3;
  params.cycles = 2;
  params.n_presample = 2;
  params.n_postsample = 2;

  Sampler sampler(std::make_shared<Operator>(op), &engine, params);

  sampler.enable_estimate_mean();
  sampler.enable_estimate_covariance();

  const std::size_t n_burnin = 1000;
  const std::size_t n_samples = 1'000'000;

  Eigen::VectorXd sample(op.size());
  sample.setZero();

  sampler.sample(sample, n_burnin);
  sampler.reset_statistics();

  sampler.sample(sample, n_samples);

  const double tol = 1e-2;
  // Expect relative error for sample covariance matrix to be near zero
  EXPECT_NEAR(1. / covariance.norm() *
                  (sampler.get_covariance() - covariance).norm(),
              0,
              tol);
}

TEST(SamplerStatisticsTest, MeanAndCovComputation) {
  namespace pg = parmgmc;

  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Vector = Eigen::VectorXd;
  using Operator = pg::LatticeOperator<Matrix, Vector>;

  auto op = std::make_shared<Operator>(2, 8, [&](const auto &lattice) {
    auto [prec, cov] = get_test_matrices(lattice, false);
    return prec;
  });

  const auto seed = 0xBEEFCAFE;
  std::mt19937 engine{seed};

  pg::GibbsSampler sampler(op, &engine);
  sampler.enable_estimate_mean();
  sampler.enable_estimate_covariance();

  std::size_t n_samples = 1000;
  std::vector<Vector> samples(n_samples);
  samples[0].resize(op->size());
  samples[0].setZero();
  sampler.sample(samples[0]);

  for (std::size_t n = 0; n < n_samples - 1; ++n) {
    samples[n + 1] = samples[n];
    sampler.sample(samples[n + 1]);
  }

  Vector zero = Vector::Zero(op->size());
  Vector mean =
      1. / n_samples * std::reduce(samples.begin(), samples.end(), zero);

  EXPECT_NEAR(sampler.get_mean().norm(), mean.norm(), 1e-12);

  Eigen::MatrixXd zero_mat(op->size(), op->size());
  zero_mat.setZero();
  Eigen::MatrixXd cov =
      std::accumulate(samples.begin(),
                      samples.end(),
                      zero_mat,
                      [&](const auto &c, const auto &sample) {
                        return c + 1. / (n_samples - 1.) * (sample - mean) *
                                       (sample - mean).transpose();
                      });

  EXPECT_NEAR(sampler.get_covariance().norm(), cov.norm(), 1e-12);
}
