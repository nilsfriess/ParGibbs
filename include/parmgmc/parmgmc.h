/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

#pragma once

#include <petsclog.h>
#include <petscmacros.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscpc.h>

#define PARMGMC_ZIGGURAT "ziggurat"

#define PCGIBBS       "gibbs"
#define PCGAMGMC      "gamgmc"
#define PCHOGWILD     "hogwild"
#define PCCHOLSAMPLER "cholsampler"
#define PCTMBSOR      "tmbsor"

typedef struct _SampleCallbackCtx {
  PetscErrorCode (*cb)(PetscInt, Vec, void *);
  void *ctx;
} *SampleCallbackCtx;

PETSC_EXTERN PetscClassId  PARMGMC_CLASSID;
PETSC_EXTERN PetscLogEvent MULTICOL_SOR;

PETSC_EXTERN PetscErrorCode ParMGMCInitialize(void);
PETSC_EXTERN PetscErrorCode PCSetSampleCallback(PC, PetscErrorCode (*)(PetscInt, Vec, void *), void *);
