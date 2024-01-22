#include <mpi.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsf.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <pcg_random.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/gibbs.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/sample_chain.hh"

#include "mat.hh"
#include "qoi.hh"

using namespace parmgmc;

template <class Chain>
inline PetscErrorCode
measure_gelman_rubin(const std::string &name, Chain &chain,
                     std::shared_ptr<GridOperator> grid_operator,
                     Vec sample_rhs) {

  PetscFunctionBeginUser;

  Vec initial_sample;
  PetscCall(DMCreateGlobalVector(grid_operator->get_dm(), &initial_sample));

  for (std::size_t i = 0; i < chain.get_n_chains(); ++i) {
    PetscCall(VecSet(initial_sample, 1000 * i));
    PetscCall(chain.set_sample(initial_sample, i));
  }
  PetscCall(VecDestroy(&initial_sample));

  PetscInt n_burnin = 50;
  PetscOptionsGetInt(NULL, NULL, "-n_burnin", &n_burnin, NULL);

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "%s: Performing burn-in...", name.c_str()));
  PetscCall(chain.sample(sample_rhs, n_burnin));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "done.\n"));
  chain.reset();

  PetscInt max_samples = 1000;
  PetscOptionsGetInt(NULL, NULL, "-max_samples", &max_samples, NULL);
  if (max_samples < 100)
    max_samples = 100;

  PetscInt run;
  auto start = std::chrono::steady_clock::now();
  for (run = 0; run < max_samples; ++run) {
    chain.sample(sample_rhs);

    // PetscCall(PetscPrintf(MPI_COMM_WORLD, "%f\n", chain.gelman_rubin()));

    if (chain.converged())
      break;
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  if (chain.converged())
    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "%s: Converged after %d steps in %ldms (R = %f)\n",
                          name.c_str(),
                          run,
                          time,
                          chain.gelman_rubin()));
  else
    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "%s: Did not converge (R = %f)\n",
                          name.c_str(),
                          chain.gelman_rubin()));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper helper(&argc, &argv);

  PetscFunctionBeginUser;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup grid operator
  int n_vertices = 5;
  PetscOptionsGetInt(NULL, NULL, "-n_vertices", &n_vertices, NULL);

  Coordinate lower_left{0, 0};
  Coordinate upper_right{1, 1};

  ColoringType coloring_type = ColoringType::RedBlack;

  auto grid_operator = std::make_shared<GridOperator>(
      n_vertices, n_vertices, lower_left, upper_right, coloring_type, assemble);

  // Setup random number generator
  pcg32 engine;
  int seed;
  if (rank == 0) {
    seed = std::random_device{}();
    PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL);
  }

  // Send seed to all other processes
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  engine.seed(seed);
  engine.set_stream(rank);

  // RHS used in samplers
  Vec sample_rhs;
  PetscCall(MatCreateVecs(grid_operator->get_mat(), NULL, &sample_rhs));
  PetscCall(fill_vec_rand(sample_rhs, engine));
  // PetscCall(VecSet(sample_rhs, 0.1));

  // Target mean ( = A^-1 * rhs)
  Vec tgt_mean;
  PetscCall(VecDuplicate(sample_rhs, &tgt_mean));
  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetOperators(ksp, grid_operator->get_mat(), grid_operator->get_mat()));
  PetscCall(KSPSolve(ksp, sample_rhs, tgt_mean));
  PetscCall(KSPDestroy(&ksp));

  ExampleQOI qoi({0, 2, 4, 6, 8, 10, 20, 40, 60, 80});

  typename ExampleQOI::DataT tgt_qoi;
  PetscCall(qoi(tgt_mean, &tgt_qoi));

  PetscInt n_chains = 16;
  PetscOptionsGetInt(NULL, NULL, "-n_chains", &n_chains, NULL);

  {
    using Chain = SampleChain<MultigridSampler<pcg32>, ExampleQOI>;

    PetscInt n_levels = 3;
    PetscOptionsGetInt(NULL, NULL, "-n_levels", &n_levels, NULL);

    PetscInt n_smooth = 1;
    PetscOptionsGetInt(NULL, NULL, "-n_smooth", &n_smooth, NULL);

    Chain chain(qoi,
                n_chains,
                tgt_mean,
                grid_operator,
                &engine,
                n_levels,
                n_smooth,
                MGMCCycleType::V,
                MGMCSmoothingType::ForwardBackward);

    PetscCall(measure_gelman_rubin("MGMC", chain, grid_operator, sample_rhs));

    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "Relative error: %f\n",
                          std::abs((chain.get_mean() - tgt_qoi) / tgt_qoi)));
  }

  {
    using Chain = SampleChain<GibbsSampler<pcg32>, ExampleQOI>;

    PetscReal omega = 1.; // SOR parameter
    PetscOptionsGetReal(NULL, NULL, "-omega", &omega, NULL);

    Chain chain(qoi,
                n_chains,
                tgt_mean,
                grid_operator,
                &engine,
                omega,
                GibbsSweepType::Symmetric);

    PetscCall(measure_gelman_rubin("Gibbs", chain, grid_operator, sample_rhs));
    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "Relative error: %f\n",
                          std::abs((chain.get_mean() - tgt_qoi) / tgt_qoi)));
  }

  PetscCall(VecDestroy(&tgt_mean));
  PetscCall(VecDestroy(&sample_rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
