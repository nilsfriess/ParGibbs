#pragma once

#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <petsc/private/pcimpl.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsystypes.h>

#include <petscoptions.h>
#include <random>

namespace parmgmc {

template <class Engine = std::mt19937> struct PCGibbs {
  PetscErrorCode init(Mat mat, Engine &engine) {
    PetscFunctionBeginUser;

    op = std::make_unique<LinearOperator>(mat, false);
    sampler = std::make_unique<MulticolorGibbsSampler<Engine>>(*op, engine);

    setup = true;

    updateSweepType();

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode setSweepType(MatSORType type) {
    PetscFunctionBeginUser;
    switch (type) {
    case SOR_FORWARD_SWEEP:
      sweepType = GibbsSweepType::Forward;
      break;
    case SOR_BACKWARD_SWEEP:
      sweepType = GibbsSweepType::Backward;
      break;
    case SOR_SYMMETRIC_SWEEP:
      sweepType = GibbsSweepType::Symmetric;
      break;
    default:
      PetscCheck(
          type == SOR_FORWARD_SWEEP || type == SOR_BACKWARD_SWEEP || type == SOR_SYMMETRIC_SWEEP,
          MPI_COMM_WORLD, PETSC_ERR_SUP, "Only SOR_{FORWARD,BACKWARD,SYMMETRIC}_SWEEP supported");
    }

    updateSweepType();

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  void updateSweepType() {
    if (setup)
      sampler->setSweepType(sweepType);
  }

  std::unique_ptr<LinearOperator> op;
  std::unique_ptr<MulticolorGibbsSampler<Engine>> sampler;

  GibbsSweepType sweepType = GibbsSweepType::Forward;

  bool setup = false;
  bool deleteEngine = false;
};

template <class Engine = std::mt19937> PetscErrorCode PCGibbsSetType(PC pc, MatSORType type) {
  PetscFunctionBeginUser;

  auto *pcg = (PCGibbs<Engine> *)pc->data;
  PetscCall(pcg->setSweepType(type));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937>
PetscErrorCode PCApplyRichardson_Gibbs(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol,
                                       PetscReal dtol, PetscInt maxits, PetscBool zeroinitialguess,
                                       PetscInt *its, PCRichardsonConvergedReason *reason) {
  PetscFunctionBeginUser;

  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)zeroinitialguess;
  (void)r;

  auto *pcdata = (PCGibbs<Engine> *)pc->data;
  if (!pcdata->setup) {
    Mat mat;
    PetscCall(PCGetOperators(pc, &mat, nullptr));

    Engine *engine = nullptr;
    PetscCall(PCGetApplicationContext(pc, &engine));

    if (!engine) {
      engine = new Engine(std::random_device{}());
      pcdata->deleteEngine = true;
    }

    pcdata->init(mat, *engine);
  }

  PetscCall(pcdata->sampler->sample(x, b, maxits));

  *its = maxits;
  *reason = PCRICHARDSON_CONVERGED_ITS;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937> PetscErrorCode PCDestroy_Gibbs(PC pc) {
  PetscFunctionBeginUser;

  auto *pcdata = (PCGibbs<Engine> *)pc->data;
  if (pcdata->deleteEngine) {
    Engine *engine = nullptr;
    PetscCall(PCGetApplicationContext(pc, &engine));
    delete engine;
  }

  delete pcdata;

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937>
PetscErrorCode PCSetFromOptions_Gibbs(PC pc, PetscOptionItems *PetscOptionsObject) {
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "Gibbs options");

  PetscCall(PetscOptionsBoolGroupBegin("-pc_gibbs_forward", "use forward Gibbs sweeps", "", &flg));
  if (flg)
    PetscCall(PCGibbsSetType<Engine>(pc, SOR_FORWARD_SWEEP));

  PetscCall(PetscOptionsBoolGroup("-pc_gibbs_symmetric", "use symmetric Gibbs sweeps", "", &flg));
  if (flg)
    PetscCall(PCGibbsSetType<Engine>(pc, SOR_SYMMETRIC_SWEEP));

  PetscCall(PetscOptionsBoolGroupEnd("-pc_gibbs_backward", "use backward Gibbs sweeps", "", &flg));
  if (flg)
    PetscCall(PCGibbsSetType<Engine>(pc, SOR_BACKWARD_SWEEP));

  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Engine = std::mt19937> PetscErrorCode PCCreate_Gibbs(PC pc) {
  PetscFunctionBeginUser;

  auto *pcgibbs = new PCGibbs<Engine>;

  pc->ops->applyrichardson = PCApplyRichardson_Gibbs<Engine>;
  pc->ops->destroy = PCDestroy_Gibbs<Engine>;
  pc->ops->setfromoptions = PCSetFromOptions_Gibbs<Engine>;

  pc->data = (void *)pcgibbs;

  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace parmgmc
