#pragma once

#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/cholesky.hh"

#include <memory>

#include <petsc/private/pcimpl.h>
#include <petscmat.h>
#include <petscsys.h>

namespace parmgmc {

template <class Engine> struct PCCholeskySampler {
  PetscErrorCode init(Mat mat, Engine &engine) {
    PetscFunctionBeginUser;

    op = std::make_unique<LinearOperator>(mat, false);
    sampler = std::make_unique<CholeskySampler<Engine>>(*op, engine);

    setup = true;

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::unique_ptr<LinearOperator> op;
  std::unique_ptr<CholeskySampler<Engine>> sampler;

  bool setup = false;
};

template <class Engine> PetscErrorCode PCApply_CholeskySampler(PC pc, Vec b, Vec x) {
  PetscFunctionBeginUser;

  auto *pcdata = (PCCholeskySampler<Engine> *)pc->data;
  if (!pcdata->setup) {
    Mat mat;
    PetscCall(PCGetOperators(pc, &mat, nullptr));

    Engine *engine;
    PetscCall(PCGetApplicationContext(pc, &engine));

    pcdata->init(mat, *engine);
  }

  PetscCall(pcdata->sampler->sample(x, b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> PetscErrorCode PCDestroy_CholeskySampler(PC pc) {
  PetscFunctionBeginUser;

  delete (PCCholeskySampler<Engine> *)pc->data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine> PetscErrorCode PCCreate_CholeskySampler(PC pc) {
  PetscFunctionBeginUser;

  auto *pcchol = new PCCholeskySampler<Engine>;

  pc->ops->apply = PCApply_CholeskySampler<Engine>;
  pc->ops->destroy = PCDestroy_CholeskySampler<Engine>;

  pc->data = (void *)pcchol;

  PetscFunctionReturn(PETSC_SUCCESS);
}

}; // namespace parmgmc
