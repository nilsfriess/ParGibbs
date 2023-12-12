#include <iostream>

#include <mpi.h>
#include <pcg_random.hpp>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmlabel.h>
#include <petscds.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/multigrid.hh"
#include "parmgmc/samplers/sample_chain.hh"
#include "parmgmc/samplers/sor.hh"

using namespace parmgmc;

PetscErrorCode assemble(Mat mat, DM dm) {
  MatStencil row_stencil;

  MatStencil col_stencil[5]; // At most 5 non-zero entries per row
  PetscScalar values[5];

  PetscFunctionBeginUser;

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(dm, &info));

  PetscReal noise_var = 1e-4;

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

      PetscCall(MatSetValuesStencil(
          mat, 1, &row_stencil, k, col_stencil, values, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetOption(mat, MAT_SYMMETRIC, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode norm_qoi(Vec sample, PetscReal *q) {
  PetscFunctionBeginUser;

  constexpr PetscInt n_vals = 5;
  PetscInt indices[n_vals] = {1, 5, 9, 11, 24};
  PetscScalar vals[n_vals];

  PetscCall(VecGetValues(sample, n_vals, indices, vals));

  *q = vals[0] + vals[1] + vals[2] + vals[3] + vals[4];

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper helper(&argc, &argv);
  PetscFunctionBeginUser;

  int n_vertices = (1 << 8) + 1;
  int n_levels = 3;
  PetscBool found;

  PetscOptionsGetInt(NULL, NULL, "-n_vertices", &n_vertices, &found);
  PetscOptionsGetInt(NULL, NULL, "-n_levels", &n_levels, &found);

  auto grid_operator =
      std::make_shared<GridOperator>(n_vertices, n_vertices, assemble);

  pcg32 engine;
  pcg_extras::seed_seq_from<std::random_device> seed_source;

  engine.seed(seed_source);
  // engine.seed(0xCAFEBEEF);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  engine.set_stream(rank);

  Vec sample;
  PetscCall(MatCreateVecs(grid_operator->mat, &sample, NULL));
  PetscCall(VecZeroEntries(sample));

  Vec rhs;
  PetscCall(VecDuplicate(sample, &rhs));
  PetscInt size;
  PetscCall(VecGetLocalSize(rhs, &size));
  PetscCall(fill_vec_rand(rhs, size, engine));
  // PetscCall(VecZeroEntries(rhs));

  KSP ksp;
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, grid_operator->mat, grid_operator->mat));
  Vec tgt_mean;
  PetscCall(VecDuplicate(sample, &tgt_mean));
  PetscCall(KSPSolve(ksp, rhs, tgt_mean));
  PetscReal tgt_mean_norm;
  PetscCall(norm_qoi(tgt_mean, &tgt_mean_norm));

  SampleChain<MultigridSampler<pcg32>> chain{
      norm_qoi, grid_operator, &engine, n_levels, assemble};
  // SampleChain<SORSampler<pcg32>> chain{
  //     norm_qoi, grid_operator, &engine, 1.9852};

  PetscInt n_samples = 1;
  PetscOptionsGetInt(NULL, NULL, "-n_samples", &n_samples, &found);

  PetscInt n_burnin = 100;
  PetscOptionsGetInt(NULL, NULL, "-n_burnin", &n_burnin, &found);
  chain.sample(sample, rhs, n_burnin);
  chain.enable_est_mean_online();
  // chain.enable_save_samples();

  for (PetscInt n = 0; n < n_samples; ++n) {
    chain.sample(sample, rhs);

    const auto mean = chain.get_mean();
    const auto rel_err = std::abs((mean - tgt_mean_norm) / tgt_mean_norm);

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "%f\n", rel_err));

    // if (rhs_norm < 1e-18)
    //   PetscCall(PetscPrintf(MPI_COMM_WORLD, "%f\n", err));
    // else
    //   std::cout << err << ", " << err / rhs_norm << "\n";
  }

  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
