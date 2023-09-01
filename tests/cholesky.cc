#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <random>

#include <pargibbs/cholesky.hh>

TEST(SamplerTest, Cholesky) {
  using namespace pargibbs;

  std::random_device rd;
  std::mt19937_64 engine{rd()};

  constexpr size_t dim = 4;
  using Matrix = Eigen::Matrix<double, dim, dim>;
  Matrix mat{{1, 0}, {0, 1}};

  CholeskySampler sampler(mat, CholeskySamplerType::CovarianceMatrix, engine);

  using Vector = Eigen::Vector<double, dim>;

  constexpr size_t n_samples = 1000000;
  std::vector<Vector> samples(n_samples);
  std::generate(samples.begin(), samples.end(),
                [&]() { return sampler.sample<Vector>(); });

  auto mean =
      (1. / n_samples) * std::reduce(samples.begin(), samples.end(), Vector(0));

  // clang-format off
  auto cov = std::transform_reduce(samples.begin(), samples.end(),
                                   Matrix{Matrix::Zero()},
                                   std::plus<Matrix>(),
                                   [&](const auto &x) {
                                     return (x - mean) * (x - mean).transpose();
                                   });
  cov *= (1. / (n_samples - 1));
  // clang-format on

  EXPECT_NEAR(mean.norm(), 0, 0.01);
  EXPECT_NEAR((cov - mat).norm(), 0, 0.01);
}
