#pragma once

#include <random>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "parmgmc/common/log.hh"

namespace parmgmc {
template <class Matrix, class Engine = std::mt19937_64> class CholeskySampler {
public:
  CholeskySampler(const Matrix &precision_matrix, Engine &engine)
      : prec(precision_matrix), engine(engine) {
    solver.compute(prec);
    if (solver.info() != Eigen::Success) {
      PARMGMC_DEBUG << "Decomposition failed.\n";
      return;
    }
  }

  template <class Vector> Vector sample(const Vector &zero) {
    Vector rand(zero);
    std::generate(rand.begin(), rand.end(), [&]() { return dist(engine); });

    return solver.matrixU().solve(rand);
  }

private:
  const Matrix &prec;
  Engine &engine;

  Eigen::SimplicialLDLT<Matrix> solver;

  std::normal_distribution<double> dist;
};
} // namespace parmgmc
