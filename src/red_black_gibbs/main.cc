#include "gmrf_operator.hh"

#include <pargibbs/pargibbs.hh>

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

#include <pcg_random.hpp>

using namespace pargibbs;

int main2() {
  Eigen::SparseVector<double> vec(10);

  vec.coeffRef(0) = 1;

  std::cout << vec << "\n";
}

int main(int argc, char *argv[]) {
  mpi_helper helper(&argc, &argv);

  pcg64 engine(0, 1 << mpi_helper::get_rank());
  if (argc > 1)
    engine.seed(std::atoi(argv[1]));
  else {
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    engine.seed(seed_source);
  }

  Lattice<2, int, LatticeOrdering::RedBlack, ParallelLayout::METIS> lattice(11);

  // if (mpi_helper::is_debug_rank()) {
  //   std::cout << "Own: ";
  //   for (auto entry : lattice.own_vertices)
  //     std::cout << entry << " ";
  //   std::cout << "\n";
  // }

  GMRFOperator precOperator(lattice);
  GibbsSampler sampler(precOperator, engine, true, 1.98);

  // if (mpi_helper::is_debug_rank())
  //   std::cout << precOperator.get_matrix() << "\n";

  using Vector = Eigen::SparseVector<double>;
  auto res = Vector(lattice.get_n_total_vertices());

  for(auto v : lattice.own_vertices) {
    res.insert(v) = 0;
    for (int n = lattice.adj_idx[v]; n < lattice.adj_idx[v+1]; ++n) {
      auto nb_idx = lattice.adj_vert[n];
      res.insert(nb_idx) = 0;
    }
  }

  const std::size_t n_burnin = 10000;
  const std::size_t n_samples = 10000;
    
  res = sampler.sample(res, n_burnin);
  sampler.reset_mean();

  const auto start = std::chrono::high_resolution_clock::now();
  res = sampler.sample(res, n_samples);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;

  double local_norm = sampler.get_mean().squaredNorm();

  double norm = 0;
  MPI_Reduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM,
             mpi_helper::debug_rank(), MPI_COMM_WORLD);
  norm = std::sqrt(norm);

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
