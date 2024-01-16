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
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/gaussian_posterior.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/gibbs.hh"
#include "parmgmc/samplers/multigrid.hh"
#include "parmgmc/samplers/sample_chain.hh"
#include "parmgmc/samplers/sor.hh"

#include "mat.hh"

using namespace parmgmc;

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

  ColoringType coloring_type = ColoringType::PETSc;

  auto grid_operator = std::make_shared<GridOperator>(
      n_vertices, n_vertices, lower_left, upper_right, coloring_type, assemble);

  // Setup random number generator
  pcg32 engine;
  std::random_device rd;
  int seed;
  if (rank == 0) {
    seed = rd();
    PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL);
  }

  // Send seed to all other processes
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  engine.seed(seed);
  engine.set_stream(rank);

  // RHS used in samplers
  Vec sample_rhs;
  PetscCall(MatCreateVecs(grid_operator->mat, NULL, &sample_rhs));
  PetscCall(fill_vec_rand(sample_rhs, engine));
  /* PetscCall(VecScale(sample_rhs, 0.1)); */
  // PetscCall(VecSet(sample_rhs, 1));

  // Target mean ( = A^-1 * rhs)
  Vec tgt_mean;
  PetscCall(VecDuplicate(sample_rhs, &tgt_mean));
  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));
  PetscCall(KSPSolve(ksp, sample_rhs, tgt_mean));
  PetscCall(KSPDestroy(&ksp));

  PetscReal tgt_norm;
  PetscCall(VecNorm(tgt_mean, NORM_2, &tgt_norm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Target norm: %f\n", tgt_norm));

  // Setup sampling chains
  // using Chain = SampleChain<GibbsSampler<pcg32>>;
  using Chain = SampleChain<MultigridSampler<pcg32>>;
  PetscInt n_chains = 1000;
  PetscOptionsGetInt(NULL, NULL, "-n_chains", &n_chains, NULL);

  PetscReal omega = 1.; // SOR parameter
  PetscOptionsGetReal(NULL, NULL, "-omega", &omega, NULL);

  // GibbsSweepType sweep_type = GibbsSweepType::FORWARD;
  std::vector<Chain> chains;
  chains.reserve(n_chains);
  for (PetscInt i = 0; i < n_chains; ++i)
    // chains.emplace_back(grid_operator, &engine, omega, sweep_type);
    chains.emplace_back(grid_operator, &engine, 5);

  PetscInt n_samples = 100;
  PetscOptionsGetInt(NULL, NULL, "-n_samples", &n_samples, NULL);

  std::vector<Vec> samples(n_chains);
  for (PetscInt i = 0; i < n_chains; ++i) {
    PetscCall(DMCreateGlobalVector(grid_operator->dm, &(samples[i])));
    PetscCall(VecSet(samples[i], 0.)); // Initial zero guess
  }

  Vec mean;
  PetscCall(VecDuplicate(samples[0], &mean));

  for (PetscInt n = 0; n < n_samples; ++n) {
    PetscCall(VecZeroEntries(mean));
    for (PetscInt c = 0; c < n_chains; ++c) {
      PetscCall(chains[c].sample(samples[c], sample_rhs));

      PetscCall(VecAXPY(mean, 1. / n_chains, samples[c]));
    }

    // Print diagonstics
    PetscReal err_norm;
    PetscCall(VecAXPY(mean, -1, tgt_mean));
    PetscCall(VecNorm(mean, NORM_2, &err_norm));

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "%f\n", err_norm / tgt_norm));
  }

  for (auto v : samples)
    PetscCall(VecDestroy(&v));

  PetscCall(VecDestroy(&tgt_mean));
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&sample_rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
