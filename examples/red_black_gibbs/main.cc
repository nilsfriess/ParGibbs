#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <pcg_random.hpp>

#include "gmrf_operator.hh"

#include "pargibbs/common/log.hh"
#include "pargibbs/lattice/lattice.hh"
#include "pargibbs/lattice/types.hh"
#include "pargibbs/mpi_helper.hh"
#include "pargibbs/samplers/gibbs.hh"

using namespace pargibbs;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char *argv[]) {
  mpi_helper helper(&argc, &argv);

  if (argc < 2) {
    std::cout << "Provide path to config file as command line argument\n";
    return 1;
  }
  std::ifstream f(argv[1]);
  json config = json::parse(f);

  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 engine(0, mpi_helper::get_rank());
  engine.seed(seed_source);

#ifdef USE_METIS
  Lattice2D lattice(11, ParallelLayout::METIS);
#else
  Lattice2D lattice(11, ParallelLayout::WORB);
#endif

  GMRFOperator prec_op(lattice);
  auto *prec_matrix = &(prec_op.matrix);

  GibbsSampler sampler(&lattice, prec_matrix, &engine, true, config["omega"]);

  using Vector = Eigen::SparseVector<double>;
  auto res = Vector(lattice.get_n_total_vertices());

  for (auto v : lattice.adj_vert)
    res.insert(v) = 0;

  const std::size_t n_burnin = config["n_burnin"];
  const std::size_t n_samples = config["n_samples"];

  res = sampler.sample(res, n_burnin);
  sampler.reset_mean();

  for (std::size_t n = 0; n < n_samples; ++n) {
    res = sampler.sample(res, 1);

    double local_norm = sampler.get_mean().squaredNorm();

    double norm = 0;
    MPI_Reduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM,
               mpi_helper::debug_rank(), MPI_COMM_WORLD);
    norm = std::sqrt(norm);

    if (mpi_helper::is_debug_rank())
      std::cout << norm << "\n";
  }

  // const auto start = std::chrono::high_resolution_clock::now();
  // res = sampler.sample(res, n_samples);
  // const auto end = std::chrono::high_resolution_clock::now();
  // const auto elapsed = end - start;

  // double local_norm = sampler.get_mean().squaredNorm();

  // double norm = 0;
  // MPI_Reduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM,
  //            mpi_helper::debug_rank(), MPI_COMM_WORLD);
  // norm = std::sqrt(norm);

  // if (mpi_helper::is_debug_rank()) {
  //   std::cout << "Norm = " << norm << "\n";
  //   std::cout << "Time: "
  //             <<
  //             std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
  //             << "\n";
  // }
}
