#include "pargibbs/common/helpers.hh"
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

  Eigen::MatrixXd exact_cov;
  // Collect all parts of the precision matrix into one dense matrix (on the
  // debug rank).
  auto full_prec = mpi_gather_matrix(prec_op.matrix);
  if (mpi_helper::is_debug_rank()) {
    exact_cov = full_prec.inverse();
  }

  std::size_t n_chains = config["n_samples"];
  const std::size_t n_samples = config["n_samples"];

  using Sampler = GibbsSampler<GMRFOperator::SparseMatrix, pcg32>;
  using Vector = Eigen::SparseVector<double>;

  std::vector<Sampler> samplers;
  std::vector<Vector> samples;
  std::vector<Eigen::VectorXd> full_samples;

  for (std::size_t i = 0; i < n_chains; ++i) {
    samplers.emplace_back(&lattice, &prec_op.matrix, &engine, config["omega"]);

    samples.emplace_back();
    samples[i].resize(lattice.get_n_total_vertices());
    for_each_ownindex_and_halo(lattice, [&](auto idx) {
      samples[i].insert(idx) = 0;
    });

    full_samples.push_back(Eigen::VectorXd(lattice.get_n_total_vertices()));
    full_samples[i].setZero();
  }

  Eigen::VectorXd mean(lattice.get_n_total_vertices());

  Eigen::MatrixXd cov(lattice.get_n_total_vertices(),
                      lattice.get_n_total_vertices());

  Eigen::SparseVector<double> prec_mean(lattice.get_n_total_vertices());
  for_each_ownindex_and_halo(lattice,
                             [&](auto idx) { prec_mean.insert(idx) = 0.123; });
  Eigen::VectorXd tgt_mean = exact_cov * prec_mean;

  for (std::size_t n = 0; n < n_samples; ++n) {
    for (std::size_t c = 0; c < n_chains; ++c) {
      samplers[c].sample(samples[c], prec_mean);

      // Remove halo values
      Vector local_sample(lattice.get_n_total_vertices());
      for (auto v : lattice.own_vertices)
        local_sample.insert(v) = samples[c].coeff(v);

      // Collect all parts of the sample which are scattered across multiple MPI
      // ranks into a single vector
      full_samples[c] = mpi_gather_vector(local_sample);
    }

    if (mpi_helper::is_debug_rank()) {
      mean.setZero();
      for (std::size_t c = 0; c < n_chains; ++c)
        mean += 1. / n_chains * full_samples[c];

      cov.setZero();
      for (std::size_t c = 0; c < n_chains; ++c)
        cov += 1. / (n_chains - 1) * (full_samples[c] - mean) *
               (full_samples[c] - mean).transpose();

      double mean_err = 1. / tgt_mean.norm() * (tgt_mean - mean).norm();
      double cov_err = 1. / exact_cov.norm() * (exact_cov - cov).norm();

      std::cout << "mean_err = " << mean_err << ", ";
      std::cout << "cov_err = " << cov_err;
      std::cout << "\n";
    }
  }
}
