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

template <class It, class Vector>
auto compute_mean(It begin, It end, const Vector &zero) {
  return std::transform_reduce(
      begin, end, zero, std::plus<Vector>(),
      [&](const auto &x) { return (1. / std::distance(begin, end)) * x; });
}

int main(int argc, char *argv[]) {
  mpi_helper helper(&argc, &argv);

  std::minstd_rand engine(std::random_device{}());

  Lattice<2, LatticeOrdering::RedBlack> lattice(301);

  GMRFOperator precOperator(lattice);
  GibbsSampler sampler(precOperator, engine, 1.9852);

  const std::size_t n_burnin = 10;
  const std::size_t n_samples = n_burnin + 1000;

  using Vector = Eigen::VectorXd;
  auto zero = Vector(lattice.get_total_points());
  zero.setZero();

  const auto start = std::chrono::steady_clock::now();
  auto res = sampler.sample(zero, n_samples);
  const auto end = std::chrono::steady_clock::now();

  if (mpi_helper::is_debug_rank()) {
    const auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PARGIBBS_DEBUG << "Time: " << time << std::endl;
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
