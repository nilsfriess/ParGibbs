#pragma once

#include <iostream>
#include <mpi.h>

#include <ostream>
#include <petsc.h>
#include <petscmat.h>

#include <utility>
#include <vector>

/** Creates finite difference matrix for the 2d Laplacian.
 */
inline Mat create_test_mat(PetscInt size_per_dim) {
  Mat mat;
  MatCreate(MPI_COMM_WORLD, &mat);
  MatSetType(mat, MATMPIAIJ);
  MatSetSizes(mat,
              PETSC_DECIDE,
              PETSC_DECIDE,
              size_per_dim * size_per_dim,
              size_per_dim * size_per_dim);
  MatMPIAIJSetPreallocation(mat, 5, nullptr, 4, nullptr);

  double h2 = 1.0 / ((size_per_dim + 1) * (size_per_dim + 1));

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
      vals.push_back(-1 / h2);
    }

    if (i < size_per_dim - 1) {
      cols.push_back(row + size_per_dim);
      vals.push_back(-1 / h2);
    }

    cols.push_back(row);
    vals.push_back(4 / h2);

    if (j > 0) {
      cols.push_back(row - 1);
      vals.push_back(-1 / h2);
    }

    if (j < size_per_dim - 1) {
      cols.push_back(row + 1);
      vals.push_back(-1 / h2);
    }

    MatSetValues(
        mat, 1, &row, cols.size(), cols.data(), vals.data(), INSERT_VALUES);
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