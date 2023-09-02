#include "pargibbs/forward_substitution.hh"
#include <cmath>
#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <random>

#include <pargibbs/cholesky.hh>
#include <pargibbs/gibbs.hh>

constexpr size_t dim = 4;
using Matrix = Eigen::Matrix<double, dim, dim>;
using Vector = Eigen::Vector<double, dim>;

Matrix get_test_covariance_mat() {
  Matrix mat{{1, 0.5, 0.2, 0.1},
             {0.5, 1, 0.3, 0.2},
             {0.2, 0.3, 1, 0.1},
             {0.1, 0.2, 0.1, 1}};
  return mat;
}

Matrix get_test_precision_mat() { return get_test_covariance_mat().inverse(); }

template <class Container> Vector compute_mean(const Container &values) {
  return (1. / values.size()) *
         std::accumulate(values.begin(), values.end(), Vector(0));
}

template <class Container> Matrix compute_cov(const Container &values) {
  const auto mean = compute_mean(values);

  // clang-format off
  auto cov = std::transform_reduce(values.begin(), values.end(),
                                   Matrix{Matrix::Zero()},
                                   std::plus<Matrix>(),
                                   [&](const auto &x) {
                                     return (x - mean) * (x - mean).transpose();
                                   });
  cov *= (1. / (values.size() - 1));
  // clang-format on
  return cov;
}

using namespace pargibbs;

TEST(SamplerTest, Cholesky) {
  std::random_device rd;
  std::mt19937_64 engine{rd()};

  const auto cov = get_test_covariance_mat();
  CholeskySampler sampler(cov, CholeskySamplerType::CovarianceMatrix, engine);

  constexpr size_t n_samples = 1000000;
  std::vector<Vector> samples(n_samples);
  std::generate(samples.begin(), samples.end(),
                [&]() { return sampler.sample<Vector>(); });

  auto sample_mean = compute_mean(samples);
  auto sample_cov = compute_cov(samples);

  constexpr double accuracy = 0.009;
  EXPECT_NEAR(sample_mean.norm(), 0, accuracy);
  EXPECT_NEAR((sample_cov - cov).norm(), 0, accuracy);
}

TEST(SamplerTest, Gibbs) {
  std::random_device rd;
  std::mt19937_64 engine{rd()};

  const auto prec = get_test_precision_mat();
  GibbsSampler sampler(prec, engine);

  constexpr size_t n_burnin = 1000;
  constexpr size_t n_samples = 1000000;
  std::vector<Vector> samples(n_burnin + n_samples);

  Vector initial(Vector::Zero());
  samples[0] = sampler.sample(initial);
  for (size_t i = 1; i < n_samples; ++i)
    samples[i] = sampler.sample(samples[i - 1]);

  // Discard burn-in samples
  samples.erase(samples.begin(), samples.begin() + n_burnin);

  auto sample_mean = compute_mean(samples);
  auto sample_cov = compute_cov(samples);

  constexpr double accuracy = 0.009;
  EXPECT_NEAR(sample_mean.norm(), 0, accuracy);
  EXPECT_NEAR((sample_cov - get_test_covariance_mat()).norm(), 0, accuracy);
}

TEST(SamplerTest, GibbsSSOR) {
  std::random_device rd;
  std::mt19937_64 engine{rd()};

  const auto prec = get_test_precision_mat();
  GibbsSampler sampler(prec, engine, 1.6);

  constexpr size_t n_burnin = 1000;
  constexpr size_t n_samples = 1000000;
  std::vector<Vector> samples(n_burnin + n_samples);

  Vector initial(Vector::Zero());
  samples[0] = sampler.sample(initial);
  for (size_t i = 1; i < n_samples; ++i)
    samples[i] = sampler.sample(samples[i - 1]);

  // Discard burn-in samples
  samples.erase(samples.begin(), samples.begin() + n_burnin);

  auto sample_mean = compute_mean(samples);
  auto sample_cov = compute_cov(samples);

  constexpr double accuracy = 0.009;
  EXPECT_NEAR(sample_mean.norm(), 0, accuracy);
  EXPECT_NEAR((sample_cov - get_test_covariance_mat()).norm(), 0, accuracy);
}

TEST(SamplerTest, GibbsWithSparseMatrix) {
  std::random_device rd;
  std::mt19937_64 engine{rd()};

  const auto cov_dense = get_test_covariance_mat();

  using Triplet = Eigen::Triplet<double>;
  // Tridiagonal part of test cov matrix
  std::vector<Triplet> triplets(dim + 2 * (dim - 1));
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      if (std::abs(i - j) <= 1)
        triplets.emplace_back(i, j, cov_dense(i, j));
    }
  }

  using SparseMatrix = Eigen::SparseMatrix<double>;
  SparseMatrix cov(dim, dim);
  cov.setFromTriplets(triplets.begin(), triplets.end());
  cov.makeCompressed();

  // Compute precision matrix for sparse cov
  Eigen::SimplicialLLT<SparseMatrix> solver;
  solver.compute(cov);

  SparseMatrix eye(dim, dim);
  eye.setIdentity();
  SparseMatrix prec = solver.solve(eye);

  GibbsSampler sampler(prec, engine);

  constexpr size_t n_burnin = 1000;
  constexpr size_t n_samples = 1000000;
  std::vector<Vector> samples(n_burnin + n_samples);

  Vector initial(Vector::Zero());
  samples[0] = sampler.sample(initial);
  for (size_t i = 1; i < n_samples; ++i)
    samples[i] = sampler.sample(samples[i - 1]);

  // Discard burn-in samples
  samples.erase(samples.begin(), samples.begin() + n_burnin);

  auto sample_mean = compute_mean(samples);
  auto sample_cov = compute_cov(samples);

  constexpr double accuracy = 0.009;
  EXPECT_NEAR(sample_mean.norm(), 0, accuracy);
  EXPECT_NEAR((sample_cov - cov).norm(), 0, accuracy);
}
