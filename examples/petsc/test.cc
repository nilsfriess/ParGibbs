#include <mpi.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <pcg_random.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <petscviewer.h>
#include <random>
#include <vector>

#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/types.hh"
#include "parmgmc/gaussian_posterior.hh"
#include "parmgmc/grid/grid_operator.hh"

using namespace parmgmc;

PetscErrorCode assemble(Mat mat, DM dm) {
  MatStencil row_stencil;

  std::array<MatStencil, 5> col_stencil; // At most 5 non-zero entries per row
  std::array<PetscScalar, 5> values;

  PetscFunctionBeginUser;

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(dm, &info));

  const PetscReal noise_var = 1e-4;

  for (auto i = info.xs; i < info.xs + info.xm; ++i) {
    for (auto j = info.ys; j < info.ys + info.ym; ++j) {
      row_stencil.i = i;
      row_stencil.j = j;

      PetscInt k = 0;

      if (i != 0) {
        values[k] = -1;
        col_stencil[k].i = i - 1;
        col_stencil[k].j = j;
        ++k;
      }

      if (i != info.mx - 1) {
        values[k] = -1;
        col_stencil[k].i = i + 1;
        col_stencil[k].j = j;
        ++k;
      }

      if (j != 0) {
        values[k] = -1;
        col_stencil[k].i = i;
        col_stencil[k].j = j - 1;
        ++k;
      }

      if (j != info.my - 1) {
        values[k] = -1;
        col_stencil[k].i = i;
        col_stencil[k].j = j + 1;
        ++k;
      }

      col_stencil[k].i = i;
      col_stencil[k].j = j;
      values[k] = static_cast<PetscScalar>(k) + noise_var;
      ++k;

      PetscCall(MatSetValuesStencil(mat,
                                    1,
                                    &row_stencil,
                                    k,
                                    col_stencil.data(),
                                    values.data(),
                                    INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetOption(mat, MAT_SYMMETRIC, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper helper(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PetscFunctionBeginUser;

  int n_vertices = (1 << 8) + 1;
  int n_levels = 3;
  PetscBool found;

  PetscOptionsGetInt(NULL, NULL, "-n_vertices", &n_vertices, &found);
  PetscOptionsGetInt(NULL, NULL, "-n_levels", &n_levels, &found);

  Coordinate lower_left{0, 0};
  Coordinate upper_right{1, 1};
  auto grid_operator = std::make_shared<GridOperator>(
      n_vertices, n_vertices, lower_left, upper_right, assemble);

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

  Vec prior_mean;
  PetscCall(MatCreateVecs(grid_operator->mat, NULL, &prior_mean));
  PetscInt size;
  PetscCall(VecGetLocalSize(prior_mean, &size));
  // PetscCall(fill_vec_rand(prior_mean, size, engine));
  PetscCall(VecSet(prior_mean, 1.));

  std::vector<Observation> obs;

  for (const auto x : {0., 0.25, 0.5, 0.75, 1.0}) {
    for (const auto y : {0., 0.25, 0.5, 0.75, 1.0}) {
      obs.push_back({{x, y}, 1.});
    }
  }

  Vec noise_diag;
  PetscCall(VecCreate(MPI_COMM_WORLD, &noise_diag));
  PetscCall(VecSetSizes(noise_diag, PETSC_DECIDE, obs.size()));
  PetscCall(VecSetUp(noise_diag));
  PetscCall(VecSet(noise_diag, 2));

  PetscInt n_chains = 10;
  PetscOptionsGetInt(NULL, NULL, "-n_chains", &n_chains, &found);

  // using Chain = SampleChain<SORSampler<pcg32>>;
  // using Chain = SampleChain<MultigridSampler<pcg32>>;
  using Sampler = GaussianPosterior<pcg32>;

  std::vector<Sampler> chains;
  chains.reserve(n_chains);
  for (PetscInt i = 0; i < n_chains; ++i)
    // chains.emplace_back(grid_operator, &engine, 1.9852);
    chains.emplace_back(grid_operator, prior_mean, obs, noise_diag, &engine);

  PetscInt n_samples = 1;
  PetscOptionsGetInt(NULL, NULL, "-n_samples", &n_samples, &found);

  std::vector<Vec> samples(n_chains);
  for (PetscInt i = 0; i < n_chains; ++i) {
    PetscCall(VecDuplicate(prior_mean, &(samples[i])));
    PetscCall(VecSet(samples[i], 1.));
  }

  Vec exact_mean;
  PetscCall(VecDuplicate(samples[0], &exact_mean));
  PetscCall(chains[0].exact_mean(exact_mean));

  // VecView(exact_mean, PETSC_VIEWER_STDOUT_WORLD);

  PetscReal exact_mean_norm;
  PetscCall(VecNorm(exact_mean, NORM_2, &exact_mean_norm));

  std::cout << "err mean norm = " << exact_mean_norm << "\n";

  Vec mean;
  PetscCall(VecDuplicate(samples[0], &mean));

  for (PetscInt n = 0; n < n_samples; ++n) {
    PetscCall(VecZeroEntries(mean));
    for (PetscInt c = 0; c < n_chains; ++c) {
      PetscCall(chains[c].sample(samples[c]));

      PetscCall(VecAXPY(mean, 1. / n_chains, samples[c]));
    }

    // PetscCall(VecAXPY(mean, -1., exact_mean));

    PetscReal err_norm;
    PetscCall(VecNorm(mean, NORM_2, &err_norm));

    // PetscCall(VecView(mean, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscPrintf(MPI_COMM_WORLD,
                          "%f, %f\n",
                          err_norm,
                          std::abs(err_norm - exact_mean_norm)));
  }

  for (auto v : samples)
    PetscCall(VecDestroy(&v));

  PetscCall(VecDestroy(&exact_mean));
  PetscCall(VecDestroy(&noise_diag));
  PetscCall(VecDestroy(&prior_mean));
  PetscCall(VecDestroy(&mean));

  PetscFunctionReturn(PETSC_SUCCESS);
}
