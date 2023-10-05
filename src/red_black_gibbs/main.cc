#include "gmrf_operator.hh"

#include <pargibbs/pargibbs.hh>

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

using namespace pargibbs;

int main(int argc, char *argv[]) {
  mpi_helper helper(&argc, &argv);

  std::mt19937 engine(std::random_device{}());

  Lattice<2, LatticeOrdering::RedBlack> lattice(11);

  GMRFOperator precOperator(lattice);
  GibbsSampler sampler(precOperator, engine, true, 1.98);

  const std::size_t n_samples = 10;
  const std::size_t n_iterations = 100;

  using Vector = Eigen::VectorXd;
  auto zero = Vector(lattice.get_total_points());
  zero.setZero();

  auto res = zero;

  // Compute cov to estimate error
  Eigen::MatrixXd dense_prec(precOperator.get_matrix());
  auto exact_cov = dense_prec.inverse();

  for (std::size_t it = 0; it < n_iterations; ++it) {
    res = sampler.sample(res, n_samples);

    if (mpi_helper::is_debug_rank()) {
      auto [mean, cov] = sampler.get_mean_cov();

      std::cout << 1. / exact_cov.norm() * (exact_cov - cov).norm() << "\n";
    }
  }

  // std::vector<Vector> samples;
  // samples.reserve(n_samples);
  // samples.push_back(sampler.sample(zero, 1));

  // for (std::size_t i = 1; i < n_samples; ++i) {
  //   samples.push_back(sampler.sample(samples.at(i - 1), 1));
  // }

  // std::cout
  //     << "Mean = "
  //     << compute_mean(samples.begin() + n_burnin, samples.end(), zero).norm()
  //     << "\n";
  // // std::cout << "||A|| = "
  // //           << Eigen::MatrixXd(precOperator.get_matrix()).operatorNorm()
  // //           << "\n";

  // CholeskySampler sampler2(precOperator.get_matrix(), engine);
  // sampler2.sample(zero);

  // samples[0] = zero;
  // for (std::size_t i = 1; i < n_samples; ++i) {
  //   samples[i] = sampler2.sample(zero);
  // }

  // std::cout
  //     << "Mean = "
  //     << compute_mean(samples.begin() + n_burnin, samples.end(), zero).norm()
  //     << "\n";
}
