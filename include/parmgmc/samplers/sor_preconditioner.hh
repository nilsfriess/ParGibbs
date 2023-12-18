#pragma once

#include "parmgmc/common/helpers.hh"

#include <iostream>
#include <mpi.h>

#include <petscerror.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscvec.h>

#include <algorithm>
#include <petscviewer.h>
#include <random>

namespace parmgmc {
template <class Engine> struct SORRichardsonContext {
  SORRichardsonContext(Engine *engine, Mat mat, PetscReal omega)
      : engine{engine}, with_lowrank_update{false} {
    PetscFunctionBeginUser;
    PetscCallVoid(init(mat, omega));
    PetscFunctionReturnVoid();
  }

  SORRichardsonContext(Engine *engine, Mat mat, Mat lowrank_factor,
                       PetscReal omega)
      : engine{engine}, lowrank_factor{lowrank_factor},
        with_lowrank_update{true} {
    PetscFunctionBeginUser;

    PetscCallVoid(init(mat, omega));

    PetscCallVoid(MatCreateVecs(lowrank_factor, &small_z, NULL));
    PetscCallVoid(VecGetLocalSize(small_z, &vec_small_size));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode init(Mat mat, PetscReal omega) {
    PetscFunctionBeginUser;

    PetscCall(MatCreateVecs(mat, &sqrt_diag, NULL));
    PetscCall(MatGetDiagonal(mat, sqrt_diag));
    PetscCall(VecSqrtAbs(sqrt_diag));
    PetscCall(VecScale(sqrt_diag, std::sqrt((2 - omega) / omega)));

    // Setup Gauss-Seidel preconditioner
    PetscCall(PCCreate(MPI_COMM_WORLD, &pc));
    PetscCall(PCSetType(pc, PCSOR));
    PetscCall(PCSetOperators(pc, mat, mat));
    PetscCall(PCSORSetOmega(pc, omega));

    PetscCall(VecDuplicate(sqrt_diag, &z));

    PetscCall(VecGetLocalSize(sqrt_diag, &vec_size));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~SORRichardsonContext() {
    VecDestroy(&small_z);
    VecDestroy(&sqrt_diag);
    VecDestroy(&z);

    PCDestroy(&pc);

    if (with_lowrank_update)
      MatDestroy(&lowrank_factor);
  }

  Engine *engine;
  std::normal_distribution<PetscReal> dist;
  Vec sqrt_diag;
  Vec z;
  PetscInt vec_size;

  Mat lowrank_factor;
  bool with_lowrank_update;
  Vec small_z;
  PetscInt vec_small_size;

  PC pc;
};

template <class Engine>
inline PetscErrorCode
sor_pc_richardson_apply(PC pc, Vec b, Vec x, Vec r, PetscReal rtol,
                        PetscReal abstol, PetscReal dtol, PetscInt maxits,
                        PetscBool zeroinitialguess, PetscInt *its,
                        PCRichardsonConvergedReason *reason) {
  /* We ignore all the provided tolerances since this is only supposed to be
   * used within MGMC
   */
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)maxits;

  // We also assume x is not zero
  (void)zeroinitialguess;

  // Always return one iteration
  *its = 1;
  *reason = PCRICHARDSON_CONVERGED_ITS;

  PetscFunctionBeginUser;

  // Get context object that contains rng and sqrt of the diagonal of the matrix
  SORRichardsonContext<Engine> *context;
  PetscCall(PCShellGetContext(pc, &context));

  // Below we set: r <- b + sqrt((2-omega)/omega) * D^1/2 * c, where c ~ N(0,I)
  PetscCall(fill_vec_rand(r, context->vec_size, *context->engine));

  PetscCall(VecPointwiseMult(r, r, context->sqrt_diag));
  PetscCall(VecAXPY(r, 1., b));

  // If operator has low-rank update, update rhs
  if (context->with_lowrank_update) {
    // Fill work vector z with N(0,1) distributed numbers
    PetscCall(fill_vec_rand(
        context->small_z, context->vec_small_size, *context->engine));

    // Multiply by given low-rank factor and add to r
    PetscCall(MatMultAdd(context->lowrank_factor, context->small_z, r, r));
  }

  // Perform actual Richardson step
  Mat A;
  PetscCall(PCGetOperators(pc, &A, NULL));

  // r <- r - A x
  PetscCall(VecScale(r, -1));
  PetscCall(MatMultAdd(A, x, r, r));
  PetscCall(VecScale(r, -1));

  // Apply (SOR) preconditioner
  PetscCall(PCApply(context->pc, r, context->z));

  PetscCall(VecAXPY(x, 1., context->z));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> PetscErrorCode sor_pc_richardson_destroy(PC pc) {
  PetscFunctionBeginUser;

  SORRichardsonContext<Engine> *context;
  PetscCall(PCShellGetContext(pc, &context));

  delete context;

  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace parmgmc
