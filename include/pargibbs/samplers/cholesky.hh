#pragma once

#include "pargibbs/common/log.hh"

#include <cstdlib>
#include <random>

#include <Eigen/SparseCholesky>

namespace pargibbs {
template <class Matrix, class Engine = std::mt19937_64> class CholeskySampler {
public:
  CholeskySampler(const Matrix &precision_matrix, Engine &engine)
      : prec(precision_matrix), engine(engine) {
    solver.compute(prec);
    if (solver.info() != Eigen::Success) {
      std::cout << "Decomposition failed.\n";
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
} // namespace pargibbs
