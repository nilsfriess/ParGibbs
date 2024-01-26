#pragma once

#include <array>

#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

inline PetscErrorCode assemble(DM dm, Mat *mat) {
  MatStencil row_stencil;

  std::array<MatStencil, 5> col_stencil; // At most 5 non-zero entries per row
  std::array<PetscScalar, 5> values;

  PetscFunctionBeginUser;

  PetscCall(DMCreateMatrix(dm, mat));

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(dm, &info));

  const PetscReal noise_var = 1e-4;

  for (auto i = info.xs; i < info.xs + info.xm; ++i) {
    for (auto j = info.ys; j < info.ys + info.ym; ++j) {
      row_stencil.i = i;
      row_stencil.j = j;

      PetscInt k = 0;

      if (i != 0) {
        values[k] = -1;
        col_stencil[k].i = i - 1;
        col_stencil[k].j = j;
        ++k;
      }

      if (i != info.mx - 1) {
        values[k] = -1;
        col_stencil[k].i = i + 1;
        col_stencil[k].j = j;
        ++k;
      }

      if (j != 0) {
        values[k] = -1;
        col_stencil[k].i = i;
        col_stencil[k].j = j - 1;
        ++k;
      }

      if (j != info.my - 1) {
        values[k] = -1;
        col_stencil[k].i = i;
        col_stencil[k].j = j + 1;
        ++k;
      }

      col_stencil[k].i = i;
      col_stencil[k].j = j;
      values[k] = static_cast<PetscScalar>(k) + noise_var;
      ++k;

      PetscCall(MatSetValuesStencil(*mat,
                                    1,
                                    &row_stencil,
                                    k,
                                    col_stencil.data(),
                                    values.data(),
                                    INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}
