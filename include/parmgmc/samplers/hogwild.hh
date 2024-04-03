#pragma once

#include "parmgmc/common/helpers.hh"
#include "parmgmc/linear_operator.hh"

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

    // // Inv sqrt diag
    // PetscCallVoid(VecDuplicate(sqrtDiag, &invSqrtDiag));
    // PetscCallVoid(VecCopy(sqrtDiag, invSqrtDiag));
    // PetscCallVoid(VecReciprocal(invSqrtDiag));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t nSamples = 1) {
    PetscFunctionBeginUser;

    if (nSamples == 0)
      PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(fillVecRand(randVec, randVecSize, engine));
    PetscCall(VecPointwiseMult(randVec, randVec, sqrtDiag));
    PetscCall(VecAXPY(randVec, 1., rhs));

    PetscCall(MatSOR(linearOperator.getMat(), randVec, 1., SOR_LOCAL_FORWARD_SWEEP, 0., nSamples,
                     1., sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  const LinearOperator &linearOperator;

  Engine &engine;

  Vec randVec;
  PetscInt randVecSize;

  // Vec invSqrtDiag;
  Vec sqrtDiag;
};
} // namespace parmgmc
