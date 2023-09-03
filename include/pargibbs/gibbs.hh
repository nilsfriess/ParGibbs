#pragma once

#include "pargibbs/common/log.hh"

#include <Eigen/Eigen>

#include <random>

namespace pargibbs {
template <class Matrix, class Engine> class GibbsSampler {
public:
  // clang-format off
  GibbsSampler(const Matrix &precisionMatrix, Engine &engine, double omega = 1)
    : precisionMatrix(precisionMatrix), omega(omega), engine(engine)
  {
    if (omega != 1) {
      PARGIBBS_DEBUG << "Sampling with omega != 1 (SSOR) is currently not implemented very efficiently\n";
    }
  }
  // clang-format on

  template <class Vector> Vector sample(const Vector &initial) {
    const auto size = precisionMatrix.rows();

    if (omega == 1) {
      const auto M = precisionMatrix.template triangularView<Eigen::Lower>();
      const auto minus_N =
          precisionMatrix.template triangularView<Eigen::StrictlyUpper>();

      Vector c;
      for (auto i = 0; i < size; ++i)
        c[i] = std::sqrt(precisionMatrix.coeff(i, i)) * norm_dist(engine);

      const auto rhs = minus_N * (-1. * initial) + c;
      return M.solve(rhs);
    } else {
      // SSOR sampling
      const auto gamma = std::sqrt(2 / omega - 1);

      auto M_full =
          Matrix(precisionMatrix.template triangularView<Eigen::Lower>());
      M_full.diagonal() /= omega;
      auto M = M_full.template triangularView<Eigen::Lower>();

      auto N_full =
          Matrix(precisionMatrix.template triangularView<Eigen::Upper>());
      N_full.diagonal() *= (omega - 1) / omega;
      N_full *= -1;
      auto N = N_full.template triangularView<Eigen::Upper>();

      Vector c1;
      std::generate(c1.begin(), c1.end(), [&]() { return norm_dist(engine); });
      Vector c2;
      std::generate(c2.begin(), c2.end(), [&]() { return norm_dist(engine); });

      const auto sqrtD =
          precisionMatrix.diagonal().array().sqrt().matrix().asDiagonal();

      const auto rhs1 = N * initial + gamma * sqrtD * c1;
      const auto x = M.solve(rhs1);

      const auto rhs2 = N.transpose() * x + gamma * sqrtD * c2;

      return M.transpose().solve(rhs2);
    }
  }

private:
  const Matrix &precisionMatrix;
  double omega;

  Engine &engine;

  std::normal_distribution<double> norm_dist;
};
}; // namespace pargibbs
