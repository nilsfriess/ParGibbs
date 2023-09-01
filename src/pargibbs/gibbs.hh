#pragma once

#include "pargibbs/common/log.hh"

#include <Eigen/Eigen>

#include <random>

namespace pargibbs {
template <class Matrix, class Engine> class GibbsSampler {
public:
  GibbsSampler(const Matrix &precisionMatrix, Engine &engine, double omega = 1)
      : precisionMatrix(precisionMatrix), omega(omega), engine(engine) {
    if (omega != 1) {
      PARGIBBS_DEBUG << "omega != 1 is not yet implemented\n";
    }
  }

  template <class Vector> Vector sample(const Vector &initial) {
    const auto size = precisionMatrix.rows();

    Vector c;
    for (auto i = 0; i < size; ++i)
      c[i] = std::sqrt(precisionMatrix(i, i)) * norm_dist(engine);

    const auto &L = precisionMatrix.template triangularView<Eigen::Lower>();
    const auto &U =
        precisionMatrix.template triangularView<Eigen::StrictlyUpper>();
    const auto rhs = U * (-1. * initial) + c;

    return L.solve(rhs);
  }

private:
  const Matrix &precisionMatrix;
  double omega;

  Engine &engine;

  std::normal_distribution<double> norm_dist;
};
}; // namespace pargibbs
