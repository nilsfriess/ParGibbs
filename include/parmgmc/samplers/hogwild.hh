#pragma once

#include "parmgmc/common/helpers.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <petscmat.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class HogwildGibbsSampler {
public:
  HogwildGibbsSampler(const LinearOperator &linearOperator, Engine &engine)
      : linearOperator{linearOperator}, engine{engine} {
    PetscFunctionBeginUser;

    PetscCallVoid(MatCreateVecs(linearOperator.getMat(), &randVec, nullptr));
    PetscCallVoid(VecGetLocalSize(randVec, &randVecSize));

    // Sqrt diag
    PetscCallVoid(VecDuplicate(randVec, &sqrtDiag));
    PetscCallVoid(MatGetDiagonal(linearOperator.getMat(), sqrtDiag));
    PetscCallVoid(VecSqrtAbs(sqrtDiag));

    PetscFunctionReturnVoid();
  }

  void setSweepType(GibbsSweepType newType) { type = newType; }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t nSamples = 1) {
    PetscFunctionBeginUser;

    if (nSamples == 0)
      PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(VecZeroEntries(randVec));
    PetscCall(fillVecRand(randVec, randVecSize, engine));
    PetscCall(VecPointwiseMult(randVec, randVec, sqrtDiag));
    PetscCall(VecAXPY(randVec, 1., rhs));

    PetscCall(MatSOR(linearOperator.getMat(), randVec, 1., gibbsSweepToPetscSweep(), 0., nSamples,
                     1., sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  MatSORType gibbsSweepToPetscSweep() {
    switch (type) {
    case GibbsSweepType::Forward:
      return SOR_LOCAL_FORWARD_SWEEP;
    case GibbsSweepType::Backward:
      return SOR_LOCAL_BACKWARD_SWEEP;
    case GibbsSweepType::Symmetric:
      return SOR_LOCAL_SYMMETRIC_SWEEP;
    default:
      return SOR_LOCAL_FORWARD_SWEEP;
    }
  }

  const LinearOperator &linearOperator;

  Engine &engine;

  Vec randVec;
  PetscInt randVecSize;

  Vec sqrtDiag;

  GibbsSweepType type = GibbsSweepType::Forward;
};
} // namespace parmgmc
