#include "parmgmc/ksp/cgs.h"
#include "parmgmc/parmgmc.h"

#include <petsc/private/kspimpl.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>

typedef struct {
  PetscRandom pr;
} *KSP_CGSampler;

static PetscErrorCode KSPDestroy_CGSampler(KSP ksp)
{
  KSP_CGSampler cgs = ksp->data;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&cgs->pr));
  PetscCall(PetscFree(cgs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetUp_CGSampler(KSP ksp)
{
  PetscFunctionBeginUser;
  PetscCall(KSPSetWorkVecs(ksp, 4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_CGSampler(KSP ksp)
{
  KSP_CGSampler cgs = ksp->data;
  Vec           x, b, r, p, w, y;
  PetscScalar   d, res;
  Mat           A, P;
  PetscInt      i;

  PetscFunctionBeginUser;
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ksp->work[0];
  p = ksp->work[1];
  w = ksp->work[2];
  y = ksp->work[3];

  PetscCall(PCGetOperators(ksp->pc, &A, &P));

  PetscCall(MatResidual(A, b, x, r));
  PetscCall(VecCopy(r, p));
  PetscCall(MatMult(A, p, w));
  PetscCall(VecDot(p, w, &d));
  PetscCall(VecCopy(x, y));

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(PCApply(ksp->pc, r, w));
    PetscCall(VecNorm(w, NORM_2, &res));
    KSPCheckNorm(ksp, res);
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNorm(r, NORM_2, &res));
    KSPCheckNorm(ksp, res);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }

  ksp->rnorm = res;
  PetscCall(KSPLogResidualHistory(ksp, res));
  PetscCall(KSPMonitor(ksp, ksp->its, res));
  PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  ksp->its = 0;
  for (i = 0; i < ksp->max_it; ++i) {
    PetscScalar g, z, rprev, rcurr, beta;

    ksp->its = i + 1;

    PetscCall(VecDot(r, r, &g));
    g = g / d;

    PetscCall(VecAXPY(x, g, p));

    PetscCall(PetscRandomGetValue(cgs->pr, &z));
    PetscCall(VecAXPY(y, z / sqrt(d), p));

    PetscCall(MatMult(A, p, w));

    PetscCall(VecDot(r, r, &rprev));
    PetscCall(VecAXPY(r, -1 * g, w));
    PetscCall(VecDot(r, r, &rcurr));
    beta = rcurr / rprev;

    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      PetscCall(PCApply(ksp->pc, r, w));
      PetscCall(VecNorm(w, NORM_2, &res));
      KSPCheckNorm(ksp, res);
      break;
    case KSP_NORM_UNPRECONDITIONED:
      PetscCall(VecNorm(r, NORM_2, &res));
      KSPCheckNorm(ksp, res);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
    }

    ksp->rnorm = res;
    PetscCall(KSPLogResidualHistory(ksp, res));
    PetscCall(KSPMonitor(ksp, ksp->its, res));
    PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));
    if (ksp->reason) break;

    PetscCall(VecAYPX(p, beta, r));
    PetscCall(PCApply(ksp->pc, r, w));
    PetscCall(VecCopy(w, r));

    PetscCall(MatMult(A, p, w));
    PetscCall(VecDot(p, w, &d));
  }
  if (i == ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscCall(VecCopy(y, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CGSamplerGetSample(KSP ksp, Vec y)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(ksp->work[3], y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPCreate_CGSampler(KSP ksp)
{
  KSP_CGSampler cgs;

  PetscFunctionBeginUser;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(PetscNew(&cgs));
  ksp->data = cgs;

  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)ksp), &cgs->pr));
  PetscCall(PetscRandomSetType(cgs->pr, PARMGMC_ZIGGURAT));
  /* PetscCall(PetscRandomSetSeed(cgs->pr, 1)); */
  /* PetscCall(PetscRandomSeed(cgs->pr)); */

  ksp->ops->setup   = KSPSetUp_CGSampler;
  ksp->ops->solve   = KSPSolve_CGSampler;
  ksp->ops->destroy = KSPDestroy_CGSampler;
  PetscFunctionReturn(PETSC_SUCCESS);
}
