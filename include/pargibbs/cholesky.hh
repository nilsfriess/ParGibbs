#pragma once

#include "pargibbs/common/log.hh"
#include <cstdlib>
#include <random>

namespace pargibbs {
enum class CholeskySamplerType { CovarianceMatrix, PrecisionMatrix };

template <class Matrix, class Engine = std::mt19937_64> class CholeskySampler {
public:
  CholeskySampler(const Matrix &matrix, CholeskySamplerType type,
                  Engine &engine)
      : matrix(matrix), type(type), engine(engine) {}

  template <class Vector> Vector sample() {
    if (!computedCholesky) {
      PARGIBBS_DEBUG
          << "Cholesky decomposition not yet computed. Computing now.\n";
      L = matrix.llt().matrixL();
      computedCholesky = true;
    }

    const auto size = L.rows();
    Vector z;

    for (auto i = 0; i < size; ++i)
      z[i] = norm_dist(engine);

    Vector res;
    if (type == CholeskySamplerType::CovarianceMatrix)
      res = L * z;
    else {
      PARGIBBS_DEBUG
          << "Cholesky sampling for precision matrix not yet implemented.";
      std::exit(-1);
    }

    return res;
  }

private:
  const Matrix &matrix;
  CholeskySamplerType type;
  Engine &engine;

  bool computedCholesky = false;
  Matrix L;

  std::normal_distribution<double> norm_dist;
};
} // namespace pargibbs
