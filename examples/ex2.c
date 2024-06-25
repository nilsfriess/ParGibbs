/*  ParMGMC - Implementation of the Multigrid Monte Carlo method in PETSc.
    Copyright (C) 2024  Nils Friess

    This file is part of ParMGMC which is released under the GNU LESSER GENERAL
    PUBLIC LICENSE (LGPL). See file LICENSE in the project root folder for full
    license details.
*/

/*  Description
 *
 *  Samples from a Gaussian random field with Matern covariance using the MS
 *  interface.
 *
 */

/**************************** Test specification ****************************/
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t -dm_refine 4
/****************************************************************************/

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscpartitioner.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <petscsys.h>

#include <parmgmc/ms.h>
#include <parmgmc/parmgmc.h>

static PetscErrorCode CreateMeshFromFilename(MPI_Comm comm, const char *filename, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateGmshFromFile(comm, filename, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define N_BURNIN  100
#define N_SAMPLES 100
#define N_SAVE    20

int main(int argc, char *argv[])
{
  DM          dm;
  MS          ms;
  Vec         x, var, mean;
  PetscViewer viewer;
  PetscBool   flag;
  char        filename[512];

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_file", filename, 512, &flag));
  if (flag) {
    PetscCall(CreateMeshFromFilename(MPI_COMM_WORLD, filename, &dm));
    PetscCall(MSSetDM(ms, dm));
  }
  flag = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-save_samples", &flag, NULL));

  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetDM(ms, &dm));

  PetscCall(DMCreateGlobalVector(dm, &x));
  for (PetscInt i = 0; i < N_BURNIN; ++i) PetscCall(MSSample(ms, x));

  if (flag) PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, "samples.vtu", FILE_MODE_WRITE, &viewer));
  PetscCall(MSBeginSaveSamples(ms, N_SAMPLES));
  for (PetscInt i = 0; i < N_SAMPLES; ++i) {
    PetscCall(MSSample(ms, x));

    if (flag && (N_SAMPLES - i - 1 < N_SAVE)) {
      char name[256];
      sprintf(name, "Sample %02d_", N_SAMPLES - i);
      PetscCall(PetscObjectSetName((PetscObject)x, name));
      PetscCall(VecView(x, viewer));
    }
  }
  PetscCall(MSEndSaveSamples(ms, N_SAMPLES));
  PetscCall(MSGetMeanAndVar(ms, &mean, &var));

  if (flag) {
    PetscCall(VecView(mean, viewer));
    PetscCall(VecView(var, viewer));

    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(MSDestroy(&ms));
  PetscCall(PetscFinalize());
  return 0;
}
