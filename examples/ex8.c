#include "parmgmc/parmgmc.h"
#include <parmgmc/ms.h>

#include <petsc.h>
#include <petsc/private/kspimpl.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscdt.h>
#include <petscfe.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscksp.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>

static PetscErrorCode CreateMeshFromFilename(MPI_Comm comm, const char *filename, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateGmshFromFile(comm, filename, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LinSpace(PetscReal start, PetscReal end, PetscInt n, PetscReal **range)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(n, range));
  for (PetscInt i = 0; i < n; ++i) (*range)[i] = start + i * (end - start) / (n - 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  Vec       b, x;
  KSP       ksp;
  MS        ms;
  Mat       A;
  DM        dm;
  PetscBool run_study = PETSC_FALSE, flag;
  char      filename[512];

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(ParMGMCInitialize());

  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_file", filename, 512, &flag));
  if (flag) {
    PetscCall(CreateMeshFromFilename(MPI_COMM_WORLD, filename, &dm));
    PetscCall(MSSetDM(ms, dm));
  }
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, &A));
  PetscCall(MSGetDM(ms, &dm));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(VecSetRandom(b, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-run_study", &run_study, NULL));
  if (run_study) {
    PetscReal     *range_threshold, *range_threshold_scale;
    PetscInt       n_range_threshold = 20, n_range_threshold_scale = 5, agg_coarse_max = 3, agg_nsmooths_max = 3;
    PetscLogDouble min_time      = INFINITY;
    PetscReal      min_threshold = -1, min_threshold_scale = -1, min_agg_coarse = -1, min_agg_nsmooth = -1;

    PetscCall(LinSpace(-.00001, 0.05, n_range_threshold, &range_threshold));
    PetscCall(LinSpace(0, 1, n_range_threshold_scale, &range_threshold_scale));
    range_threshold[0] = -1;

    for (PetscInt m = 0; m < agg_nsmooths_max; ++m) {
      for (PetscInt k = 0; k < agg_coarse_max; ++k) {
        for (PetscInt i = 0; i < n_range_threshold; ++i) {
          for (PetscInt j = 0; j < n_range_threshold_scale; ++j) {
            char           buffer[64];
            PetscLogDouble start, end;

            /* PetscCall(PetscOptionsClear(NULL)); */
            PetscCall(PetscOptionsSetValue(NULL, "-ksp_type", "richardson"));
            PetscCall(PetscOptionsSetValue(NULL, "-pc_type", "gamg"));
            PetscCall(PetscOptionsSetValue(NULL, "-mg_levels_pc_type", "sor"));
            PetscCall(PetscOptionsSetValue(NULL, "-mg_levels_ksp_type", "richardson"));
            PetscCall(PetscOptionsSetValue(NULL, "-mg_levels_ksp_max_it", "1"));
            PetscCall(PetscOptionsSetValue(NULL, "-mg_levels_pc_sor_symmetric", ""));
            /* PetscCall(PetscOptionsSetValue(NULL, "-ksp_max_it", "100")); */
            PetscCall(PetscOptionsSetValue(NULL, "-ksp_rtol", "1e-12"));
            /* PetscCall(PetscOptionsSetValue(NULL, "-pc_mg_type", "full")); */

            snprintf(buffer, sizeof buffer, "%f", range_threshold[i]);
            PetscCall(PetscOptionsSetValue(NULL, "-pc_gamg_threshold", buffer));

            snprintf(buffer, sizeof buffer, "%f", range_threshold_scale[j]);
            PetscCall(PetscOptionsSetValue(NULL, "-pc_gamg_threshold_scale", buffer));

            snprintf(buffer, sizeof buffer, "%d", k);
            PetscCall(PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_coarsening", buffer));

            snprintf(buffer, sizeof buffer, "%d", m);
            PetscCall(PetscOptionsSetValue(NULL, "-pc_gamg_agg_nsmooths", buffer));

            PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
            PetscCall(KSPSetOperators(ksp, A, A));
            PetscCall(KSPSetFromOptions(ksp));
            PetscCall(KSPSetUp(ksp));
            start = MPI_Wtime();
            PetscCall(KSPSolve(ksp, b, x));
            end = MPI_Wtime();
            PetscCall(PetscPrintf(MPI_COMM_WORLD, "Agg. nsmooth %d, Agg. Coarsening %d, Threshold %.5f, Scale %.5f: %.6f\n", m, k, range_threshold[i], range_threshold_scale[j], (end - start) * 1000));

            if (end - start < min_time) {
              min_time            = end - start;
              min_threshold       = range_threshold[i];
              min_threshold_scale = range_threshold_scale[j];
              min_agg_coarse      = k;
              min_agg_nsmooth     = m;
            }

            PetscCall(KSPDestroy(&ksp));

            if (range_threshold[i] == -1) j = n_range_threshold_scale; // For range_threshold == -1 we don't need to scan anything
          }
        }
      }
    }

    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Minimum runtime %.5f s with:\n", min_time));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t-pc_gamg_threshold %f ", min_threshold));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t-pc_gamg_threshold_scale %f ", min_threshold_scale));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t-pc_gamg_aggressive_coarsening %.0f ", min_agg_coarse));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "\t-pc_gamg_agg_nsmooths %.0f", min_agg_nsmooth));
  } else {
    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetDM(ksp, dm));
    PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetUp(ksp));

    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPDestroy(&ksp));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MSDestroy(&ms));
  PetscCall(PetscFinalize());
  return 0;
}
