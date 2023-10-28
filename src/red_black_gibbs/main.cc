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

int main(int argc, char *argv[]) {
  mpi_helper helper(&argc, &argv);

  pcg64 engine(0, 1 << mpi_helper::get_rank());
  if (argc > 1)
    engine.seed(std::atoi(argv[1]));
  else {
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    engine.seed(seed_source);
  }

#ifdef USE_METIS
  Lattice<2, int, LatticeOrdering::RedBlack, ParallelLayout::METIS> lattice(21);
#else
  Lattice<2, std::size_t, LatticeOrdering::RedBlack, ParallelLayout::WORB>
      lattice(21);
#endif

  GMRFOperator precOperator(lattice);
  GibbsSampler sampler(precOperator, engine, true, 1.68);

  using Vector = Eigen::SparseVector<double>;
  auto res = Vector(lattice.get_n_total_vertices());

  for (auto v : lattice.adj_vert)
    res.insert(v) = 0;

  const std::size_t n_burnin = 10000;
  const std::size_t n_samples = 50000;

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
}
