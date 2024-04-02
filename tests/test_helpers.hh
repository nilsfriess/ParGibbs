#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <mpi.h>

#include <ostream>
#include <petsc.h>
#include <petscao.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscmat.h>

#include <petscsystypes.h>
#include <petscviewer.h>
#include <utility>
#include <vector>

inline Mat create_test_mat(PetscInt size_per_dim) {

  Mat mat;
  MatCreate(MPI_COMM_WORLD, &mat);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 1)
    MatSetType(mat, MATMPIAIJ);
  else
    MatSetType(mat, MATSEQAIJ);
  MatSetSizes(mat,
              PETSC_DECIDE,
              PETSC_DECIDE,
              size_per_dim * size_per_dim,
              size_per_dim * size_per_dim);

  if (size > 1)
    MatMPIAIJSetPreallocation(mat, 5, nullptr, 4, nullptr);
  else
    MatSeqAIJSetPreallocation(mat, 5, nullptr);

  PetscInt local_start, local_end;
  MatGetOwnershipRange(mat, &local_start, &local_end);

  const auto index_to_grid = [size_per_dim](auto row) {
    return std::make_pair(row / size_per_dim, row % size_per_dim);
  };

  std::vector<PetscInt> cols;
  std::vector<PetscReal> vals;

  for (PetscInt row = local_start; row < local_end; ++row) {
    const auto [i, j] = index_to_grid(row);
    cols.clear();
    vals.clear();

    if (i > 0) {
      cols.push_back(row - size_per_dim);
      vals.push_back(-1);
    }

    if (i < size_per_dim - 1) {
      cols.push_back(row + size_per_dim);
      vals.push_back(-1);
    }

    cols.push_back(row);
    vals.push_back(6);

    if (j > 0) {
      cols.push_back(row - 1);
      vals.push_back(-1);
    }

    if (j < size_per_dim - 1) {
      cols.push_back(row + 1);
      vals.push_back(-1);
    }

    MatSetValues(
        mat, 1, &row, cols.size(), cols.data(), vals.data(), INSERT_VALUES);
  }

  MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

  MatSetOption(mat, MAT_SPD, PETSC_TRUE);

  return mat;
}

inline std::pair<Mat, std::vector<PetscInt>>
create_test_mat(DM dm, double kappainv = 1.) {
  const auto kappa2 = (1. / kappainv) * (1. / kappainv);

  DMDALocalInfo info;
  DMDAGetLocalInfo(dm, &info);

  assert(info.dim == 2 && "Only dim = 2 supported currently");
  assert(info.mx == info.my && "Only square DMDA supported currently");

  Mat mat;
  DMCreateMatrix(dm, &mat);

  MatSetOption(mat, MAT_USE_INODES, PETSC_FALSE);

  MatStencil row;
  std::array<MatStencil, 5> cols;
  std::array<PetscReal, 5> vals;

  double h2inv = 1. / ((info.mx - 1) * (info.mx - 1));

  std::vector<PetscInt> dirichletRows;
  dirichletRows.reserve(4 * info.mx);

  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      row.j = j;
      row.i = i;

      if ((i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1)) {
        dirichletRows.push_back(j * info.my + i);
      } else {
        std::size_t k = 0;

        if (j != 0) {
          cols[k].j = j - 1;
          cols[k].i = i;
          vals[k] = -h2inv;
          ++k;
        }

        if (i != 0) {
          cols[k].j = j;
          cols[k].i = i - 1;
          vals[k] = -h2inv;
          ++k;
        }

        cols[k].j = j;
        cols[k].i = i;
        vals[k] = 4 * h2inv + kappa2;
        ++k;

        if (j != info.my - 1) {
          cols[k].j = j + 1;
          cols[k].i = i;
          vals[k] = -h2inv;
          ++k;
        }

        if (i != info.mx - 1) {
          cols[k].j = j;
          cols[k].i = i + 1;
          vals[k] = -h2inv;
          ++k;
        }

        MatSetValuesStencil(
            mat, 1, &row, k, cols.data(), vals.data(), INSERT_VALUES);
      }
    }
  }

  MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

  // Dirichlet rows are in natural ordering, convert to global using the DM's AO
  AO ao;
  DMDAGetAO(dm, &ao);
  AOApplicationToPetsc(ao, dirichletRows.size(), dirichletRows.data());

  MatSetOption(mat, MAT_SPD, PETSC_TRUE);

  return {mat, dirichletRows};
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
               nullptr,
               nullptr,
               &dm);

  DMSetUp(dm);
  DMDASetUniformCoordinates(
      dm, lower_left.x, upper_right.x, lower_left.y, upper_right.y, 0, 0);

  return dm;
}

inline bool operator==(PetscSFNode n1, PetscSFNode n2) {
  return n1.index == n2.index && n1.rank == n2.rank;
}

inline std::ostream &operator<<(std::ostream &out, const PetscSFNode &node) {
  out << "[" << node.rank << "]->" << node.index;
  return out;
}