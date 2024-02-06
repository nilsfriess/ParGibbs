#pragma once

#include "petscviewer.h"
#include <iostream>
#include <mpi.h>
#include <petsc.h>
#include <petscmat.h>

#include <vector>

inline Mat create_test_mat(PetscInt size, bool no_off_diag = false) {
  Mat mat;
  MatCreateAIJ(MPI_COMM_WORLD,
               PETSC_DECIDE,
               PETSC_DECIDE,
               size,
               size,
               3,
               nullptr,
               no_off_diag ? 0 : 2,
               nullptr,
               &mat);
  MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  PetscInt local_start, local_end;
  MatGetOwnershipRange(mat, &local_start, &local_end);
  auto local_size = local_end - local_start;

  PetscInt min_local_size;
  MPI_Allreduce(
      &local_size, &min_local_size, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  PetscInt n_cols = no_off_diag ? 3 : 5;

  std::vector<PetscInt> cols(n_cols);
  std::vector<PetscReal> vals(n_cols);

  for (PetscInt row = local_start; row < local_end; ++row) {
    PetscInt nz = 0;
    if (not no_off_diag && row - min_local_size >= 0) {
      cols[nz] = row - min_local_size;
      vals[nz] = -2;
      nz++;
    }

    if (row - 1 >= 0) {
      cols[nz] = row - 1;
      vals[nz] = -1;
      nz++;
    }

    cols[nz] = row;
    vals[nz] = 8;
    nz++;

    if (row + 1 < size) {
      cols[nz] = row + 1;
      vals[nz] = -1;
      nz++;
    }

    if (not no_off_diag && row + min_local_size < size) {
      cols[nz] = row + min_local_size;
      vals[nz] = -2;
      nz++;
    }

    MatSetValues(mat, 1, &row, nz, cols.data(), vals.data(), INSERT_VALUES);
  }

  MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

  return mat;
}


struct Coordinate {
  PetscReal x;
  PetscReal y;
};

inline DM create_test_dm(PetscInt n_vertices_per_dim) {
  Coordinate lower_left{0, 0};
  Coordinate upper_right{1, 1};

  PetscInt n_vertices = n_vertices_per_dim;
  PetscInt dof_per_node = 1;
  PetscInt stencil_width = 1;

  DM dm;
  DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE,
               DM_BOUNDARY_NONE,
               DMDA_STENCIL_STAR,
               n_vertices,
               n_vertices,
               PETSC_DECIDE,
               PETSC_DECIDE,
               dof_per_node,
               stencil_width,
               NULL,
               NULL,
               &dm);

  DMSetUp(dm);
  DMDASetUniformCoordinates(
      dm, lower_left.x, upper_right.x, lower_left.y, upper_right.y, 0, 0);

  return dm;
}
