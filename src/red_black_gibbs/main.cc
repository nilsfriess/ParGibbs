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

  // TODO: Use a proper parallel rng
  // std::mt19937 engine(12345 + mpi_helper::get_rank());
  std::mt19937 engine(std::random_device{}());

  Lattice<2, LatticeOrdering::RedBlack, ParallelLayout::WORB> lattice(17);

  GMRFOperator precOperator(lattice);
  GibbsSampler sampler(precOperator, engine, true, 1.98);

  const std::size_t n_samples = 10000;

  using Vector = Eigen::VectorXd;
  auto zero = Vector(lattice.get_n_total_vertices());
  zero.setZero();

  const std::size_t n_burnin = 1000;
  sampler.sample(zero, n_burnin);
  sampler.reset_mean();

  auto res = zero;
  const auto start = std::chrono::high_resolution_clock::now();
  res = sampler.sample(res, n_samples);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;

  double local_norm = 0;
  const auto s_mean = sampler.get_mean();
  for (std::size_t v = 0; v < lattice.get_n_total_vertices(); ++v) {
    if (lattice.mpiowner[v] != mpi_helper::get_rank())
      continue;
    const auto coeff = s_mean.coeff(v);
    local_norm += std::abs(coeff);
  }

  double norm = local_norm;
  MPI_Reduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM,
             mpi_helper::debug_rank(), MPI_COMM_WORLD);

  if (mpi_helper::is_debug_rank()) {
    std::cout << "Norm = " << norm << "\n";
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
              << "\n";
  }

  // const auto [size, rank] = mpi_helper::get_size_rank();
  // for (auto index : indices)
  //   std::cout << rank << ": " << mean[index] << "\n";

  // Int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype
  // datatype,
  //                MPI_Op op, int root, MPI_Comm comm)

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
