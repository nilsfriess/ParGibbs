#pragma once

#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <petsc/private/pcimpl.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsystypes.h>

namespace parmgmc {

template <class Engine> struct PCGibbs {
  PetscErrorCode init(Mat mat, Engine &engine) {
    PetscFunctionBeginUser;

    op = std::make_unique<LinearOperator>(mat, false);
    sampler = std::make_unique<MulticolorGibbsSampler<Engine>>(*op, engine);

    setup = true;

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::unique_ptr<LinearOperator> op;
  std::unique_ptr<MulticolorGibbsSampler<Engine>> sampler;

  bool setup = false;
};

template <class Engine>
PetscErrorCode PCApplyRichardson_Gibbs(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol,
                                       PetscReal dtol, PetscInt maxits, PetscBool zeroinitialguess,
                                       PetscInt *its, PCRichardsonConvergedReason *reason) {
  PetscFunctionBeginUser;

  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)zeroinitialguess;
  (void)r;

  Mat mat;
  PetscCall(PCGetOperators(pc, &mat, nullptr));

  Engine *engine;
  PetscCall(PCGetApplicationContext(pc, &engine));

  auto *pcdata = (PCGibbs<Engine> *)pc->data;
  if (!pcdata->setup)
    pcdata->init(mat, *engine);

  PetscCall(pcdata->sampler->sample(x, b, maxits));

  *its = maxits;
  *reason = PCRICHARDSON_CONVERGED_ITS;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> PetscErrorCode PCDestroy_Gibbs(PC pc) {
  PetscFunctionBeginUser;

  delete (PCGibbs<Engine> *)pc->data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> PetscErrorCode PCCreate_Gibbs(PC pc) {
  PetscFunctionBeginUser;

  auto *pcgibbs = new PCGibbs<Engine>;

  pc->ops->applyrichardson = PCApplyRichardson_Gibbs<Engine>;
  pc->ops->destroy = PCDestroy_Gibbs<Engine>;

  pc->data = (void *)pcgibbs;

  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace parmgmc
