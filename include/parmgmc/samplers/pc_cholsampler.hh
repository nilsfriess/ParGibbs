#pragma once

#include <petsc/private/pcimpl.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <random>

#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO

#include "parmgmc/samplers/cholesky.hh"

#include "parmgmc/linear_operator.hh"
#include <memory>

namespace parmgmc {

template <class Engine = std::mt19937> struct PCCholeskySampler {
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
  bool deleteEngine = false;
};

template <class Engine = std::mt19937> PetscErrorCode PCApply_CholeskySampler(PC pc, Vec b, Vec x) {
  PetscFunctionBeginUser;

  auto *pcdata = (PCCholeskySampler<Engine> *)pc->data;
  if (!pcdata->setup) {
    Mat mat;
    PetscCall(PCGetOperators(pc, &mat, nullptr));

    Engine *engine;
    PetscCall(PCGetApplicationContext(pc, &engine));

    if (!engine) {
      engine = new Engine(std::random_device{}());
      pcdata->deleteEngine = true;
    }

    pcdata->init(mat, *engine);
  }

  PetscCall(pcdata->sampler->sample(x, b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937> PetscErrorCode PCDestroy_CholeskySampler(PC pc) {
  PetscFunctionBeginUser;

  auto *pcdata = (PCCholeskySampler<Engine> *)pc->data;
  if (pcdata->deleteEngine) {
    Engine *engine = nullptr;
    PetscCall(PCGetApplicationContext(pc, &engine));
    delete engine;
  }

  delete pcdata;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937> PetscErrorCode PCCreate_CholeskySampler(PC pc) {
  PetscFunctionBeginUser;

  auto *pcchol = new PCCholeskySampler<Engine>;

  pc->ops->apply = PCApply_CholeskySampler<Engine>;
  pc->ops->destroy = PCDestroy_CholeskySampler<Engine>;

  pc->data = (void *)pcchol;

  PetscFunctionReturn(PETSC_SUCCESS);
}

}; // namespace parmgmc

#else
namespace parmgmc {

template <class Engine = std::mt19937> struct PCCholeskySampler {
  Vec r = nullptr;
  Vec v = nullptr;

  bool deleteEngine = false;
  Engine *engine;

  Mat factor;
};

template <class Engine = std::mt19937> PetscErrorCode PCApply_CholeskySampler(PC pc, Vec b, Vec x) {
  PetscFunctionBeginUser;

  auto *pcdata = (PCCholeskySampler<Engine> *)pc->data;
  if (pcdata->v == nullptr) {
    Mat mat;
    PetscCall(PCGetOperators(pc, &mat, nullptr));
    PetscCall(MatGetFactor(mat, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &pcdata->factor));

    IS rowperm, colperm;
    PetscCall(MatGetOrdering(mat, MATORDERINGNATURAL, &rowperm, &colperm));

    MatFactorInfo info;

    PetscCall(MatCholeskyFactorSymbolic(pcdata->factor, mat, rowperm, &info));
    PetscCall(MatCholeskyFactorNumeric(pcdata->factor, mat, &info));

    PetscCall(ISDestroy(&rowperm));
    PetscCall(ISDestroy(&colperm));

    PetscCall(VecDuplicate(b, &pcdata->v));
    PetscCall(VecDuplicate(b, &pcdata->r));

    Engine *engine;
    PetscCall(PCGetApplicationContext(pc, &engine));

    if (!engine) {
      pcdata->engine = new Engine(std::random_device{}());
      pcdata->deleteEngine = true;
    }
  }

  PetscCall(fillVecRand(pcdata->r, *pcdata->engine));
  PetscCall(MatForwardSolve(pcdata->factor, b, pcdata->v));

  PetscCall(VecAXPY(pcdata->v, 1., pcdata->r));

  PetscCall(MatBackwardSolve(pcdata->factor, pcdata->v, x));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937> PetscErrorCode PCDestroy_CholeskySampler(PC pc) {
  PetscFunctionBeginUser;

  auto *pcdata = (PCCholeskySampler<Engine> *)pc->data;
  if (pcdata->deleteEngine) {
    Engine *engine = nullptr;
    PetscCall(PCGetApplicationContext(pc, &engine));
    delete engine;
  }

  delete pcdata;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937> PetscErrorCode PCCreate_CholeskySampler(PC pc) {
  PetscFunctionBeginUser;

  auto *pcchol = new PCCholeskySampler<Engine>;

  pc->ops->apply = PCApply_CholeskySampler<Engine>;
  pc->ops->destroy = PCDestroy_CholeskySampler<Engine>;

  pc->data = (void *)pcchol;

  PetscFunctionReturn(PETSC_SUCCESS);
}

}; // namespace parmgmc

#endif
