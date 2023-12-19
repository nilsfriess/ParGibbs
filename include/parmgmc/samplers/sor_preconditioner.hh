#pragma once

#include "parmgmc/common/helpers.hh"
#include "parmgmc/grid/grid_operator.hh"

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
  SORRichardsonContext(std::shared_ptr<GridOperator> op, Engine *engine,
                       PetscReal omega)
      : op{op}, engine{engine} {
    PetscFunctionBeginUser;

    PetscCallVoid(MatCreateVecs(op->mat, &sqrt_diag, NULL));
    PetscCallVoid(MatGetDiagonal(op->mat, sqrt_diag));
    PetscCallVoid(VecSqrtAbs(sqrt_diag));
    PetscCallVoid(VecScale(sqrt_diag, std::sqrt((2 - omega) / omega)));

    // Setup Gauss-Seidel preconditioner
    PetscCallVoid(PCCreate(MPI_COMM_WORLD, &pc));
    PetscCallVoid(PCSetType(pc, PCSOR));
    PetscCallVoid(PCSORSetSymmetric(pc, SOR_LOCAL_FORWARD_SWEEP));
    PetscCallVoid(PCSetOperators(pc, op->mat, op->mat));
    PetscCallVoid(PCSORSetOmega(pc, omega));

    // Setup (big) work vector
    PetscCallVoid(VecDuplicate(sqrt_diag, &z));
    PetscCallVoid(VecGetLocalSize(sqrt_diag, &vec_size));

    // Setup (small) work vector
    if (op->lowrank_update) {
      PetscCallVoid(op->lowrank_update->create_compatible_vecs(NULL, &small_z));
      PetscCallVoid(VecGetLocalSize(small_z, &vec_small_size));
    }

    PetscFunctionReturnVoid();
  }

  ~SORRichardsonContext() {
    VecDestroy(&small_z);
    VecDestroy(&sqrt_diag);
    VecDestroy(&z);

    PCDestroy(&pc);
  }

  std::shared_ptr<GridOperator> op;

  Engine *engine;
  std::normal_distribution<PetscReal> dist;
  Vec sqrt_diag;
  Vec z;
  PetscInt vec_size;

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

  // r <- b + sqrt((2-omega)/omega) * D^1/2 * c, where c ~ N(0,I)
  PetscCall(fill_vec_rand(r, context->vec_size, *context->engine));
  PetscCall(VecPointwiseMult(r, r, context->sqrt_diag));

  if (context->op->lowrank_update) {
    PetscCall(fill_vec_rand(
        context->small_z, context->vec_small_size, *context->engine));

    // Multiply by given low-rank factor and add to r
    PetscCall(context->op->lowrank_update->apply_cholesky_L(context->small_z,
                                                            context->z));
    PetscCall(VecAXPY(r, -1., context->z));
  }

  PetscCall(VecAXPY(r, 1., b));

  // Perform actual Richardson step
  Mat A;
  PetscCall(PCGetOperators(pc, &A, NULL));

  // r <- r - A x
  PetscCall(context->op->apply(x, context->z));
  PetscCall(VecAXPY(r, -1., context->z));

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
