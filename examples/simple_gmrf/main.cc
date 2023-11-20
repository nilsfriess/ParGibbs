#include <Eigen/Eigen>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

#include <pcg_random.hpp>

#include "gmrf_operator.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/lattice_operator.hh"
#include "parmgmc/common/log.hh"
#include "parmgmc/lattice/lattice.hh"
#include "parmgmc/lattice/types.hh"
#include "parmgmc/mpi_helper.hh"
#include "parmgmc/samplers/gibbs.hh"
#include "parmgmc/samplers/multigrid.hh"

using namespace parmgmc;

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

  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Operator = LatticeOperator<Matrix, Vector>;

  auto op = std::make_shared<Operator>(2, 9, gmrf_matrix_builder);
  for (auto v : op->get_lattice().own_vertices)
    op->vector().coeffRef(v) = 1;

  Eigen::MatrixXd exact_cov;
  // Collect all parts of the precision matrix into one dense matrix (on the
  // debug rank).
  auto full_prec = mpi_gather_matrix(op->get_matrix());
  if (mpi_helper::is_debug_rank()) {
    exact_cov = full_prec.inverse();
  }

  std::size_t n_chains = config["n_chains"];
  const std::size_t n_samples = config["n_samples"];

  // using Sampler = GibbsSampler<Operator, pcg32>;
  using Sampler = MultigridSampler<Operator, pcg32>;

  std::vector<Sampler> samplers;
  std::vector<Vector> samples;
  std::vector<Eigen::VectorXd> full_samples;

  Sampler::Parameters params;
  params.levels = 3;
  params.cycles = 1;
  params.n_presample = 4;
  params.n_postsample = 0;

  for (std::size_t i = 0; i < n_chains; ++i) {
    // samplers.emplace_back(op, &engine, config["omega"]);
    samplers.emplace_back(op, &engine, params);

    samples.emplace_back(op->size());
    for_each_ownindex_and_halo(op->get_lattice(),
                               [&](auto idx) { samples[i].coeffRef(idx) = 0; });

    full_samples.emplace_back(op->size());
    full_samples[i].setZero();
  }

  Eigen::VectorXd mean(op->size());
  Eigen::MatrixXd cov(op->size(), op->size());

  Eigen::VectorXd tgt_mean;
  if (mpi_helper::is_debug_rank())
    tgt_mean = exact_cov * op->vector();

  for (std::size_t n = 0; n < n_samples; ++n) {
    for (std::size_t c = 0; c < n_chains; ++c) {
      samplers[c].sample(samples[c]);

      // Remove halo values
      Eigen::VectorXd local_sample(op->size());
      for (auto v : op->get_lattice().own_vertices)
        local_sample[v] = samples[c].coeff(v);

      // Collect all parts of the sample which are scattered across multiple MPI
      // ranks into a single vector
      full_samples[c] = mpi_gather_vector(local_sample, op->get_lattice());
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
