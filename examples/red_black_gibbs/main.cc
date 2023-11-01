#include <Eigen/Eigen>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

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

  pcg32 engine(0, mpi_helper::get_rank());

  if (argc > 2)
    engine.seed(std::atoi(argv[2]));
  else {
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    engine.seed(seed_source);
  }

#ifdef USE_METIS
  Lattice lattice(config["dim"], config["lattice_size"], ParallelLayout::METIS);
#else
  Lattice lattice(config["dim"], config["lattice_size"], ParallelLayout::WORB);
#endif

  GMRFOperator prec_op(lattice);
  auto *prec_matrix = &(prec_op.matrix);

  auto dense_prec = Eigen::MatrixXd(prec_op.matrix);
  auto exact_cov = dense_prec.inverse();

  std::size_t n_chains = 10000;
  // const std::size_t n_burnin = config["n_burnin"];
  const std::size_t n_samples = config["n_samples"];

  using Sampler = GibbsSampler<GMRFOperator::SparseMatrix, pcg32>;
  using Vector = Eigen::SparseVector<double>;

  std::vector<Sampler> samplers;
  std::vector<Vector> samples;
  samplers.reserve(10);
  samples.reserve(10);

  for (std::size_t i = 0; i < n_chains; ++i)
    samplers.emplace_back(&lattice, prec_matrix, &engine, config["omega"]);

  for (std::size_t i = 0; i < n_chains; ++i) {
    samples.push_back(Vector(lattice.get_n_total_vertices()));
    samples[i].setZero();
  }

  Vector mean(lattice.get_n_total_vertices());
  Eigen::MatrixXd cov(lattice.get_n_total_vertices(),
                      lattice.get_n_total_vertices());

  for (std::size_t n = 0; n < n_samples; ++n) {
    for (std::size_t c = 0; c < n_chains; ++c)
      samplers[c].sample(samples[c]);

    mean.setZero();
    for (std::size_t c = 0; c < n_chains; ++c)
      mean += 1. / n_chains * samples[c];

    cov.setZero();
    for (std::size_t c = 0; c < n_chains; ++c)
      cov += 1. / (n_chains - 1) * (samples[c] - mean) *
             (samples[c] - mean).transpose();

    double err = 1. / exact_cov.norm() * (exact_cov - cov).norm();

    std::cout << "mean_err = " << mean.norm() << ", ";
    std::cout << "cov_err = " << err;
    std::cout << "\n";
  }
}
