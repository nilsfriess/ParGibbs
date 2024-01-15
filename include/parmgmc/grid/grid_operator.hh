#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <petscao.h>
#include <petscis.h>
#include <vector>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
/* Represents a low rank matrix of the form A S A^T.
   The template parameter can either be Vec, if the matrix S is a diagonal
   matrix or Mat if it is a full matrix. */
template <typename MiddleMat = Mat> class LowrankUpdate {
  static_assert(std::is_same_v<MiddleMat, Mat> ||
                std::is_same_v<MiddleMat, Vec>);

public:
  LowrankUpdate(Mat A, MiddleMat S, MiddleMat S_chol)
      : A{A}, S{S}, S_chol{S_chol} {
    PetscFunctionBeginUser;

    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCallVoid(VecDuplicate(S, &z));
    } else {
      PetscCallVoid(MatCreateVecs(S, &z, NULL));
    }

    PetscFunctionReturnVoid();
  }

  ~LowrankUpdate() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&z));
    PetscCallVoid(MatDestroy(&A));
    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCallVoid(VecDestroy(&S));
      PetscCallVoid(VecDestroy(&S_chol));
    } else {
      PetscCallVoid(MatDestroy(&S));
      PetscCallVoid(MatDestroy(&S_chol));
    }

    PetscFunctionReturnVoid();
  }

  PetscErrorCode apply(Vec xin, Vec xout) const {
    PetscFunctionBeginUser;

    PetscCall(MatMult(A, xin, z));

    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCall(VecPointwiseMult(z, z, S));
    } else {
      // PetscCall(MatMult(S, xout, z));
      static_assert(!std::is_same_v<MiddleMat, Mat>, "Not implemented yet");
    }

    PetscCall(MatMultTranspose(A, z, xout));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode apply_cholesky_L(Vec xin, Vec xout) const {
    PetscFunctionBeginUser;

    if constexpr (std::is_same_v<MiddleMat, Vec>) {
      PetscCall(VecPointwiseMult(z, xin, S_chol));
    } else {
      // PetscCall(MatMult(S_chol, xout, z));
      // PetscCall(VecCopy(z, xout));
      static_assert(!std::is_same_v<MiddleMat, Mat>, "Not implemented yet");
    }

    PetscCall(MatMultTranspose(A, z, xout));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode create_compatible_vecs(Vec *big, Vec *small) const {
    PetscFunctionBeginUser;

    if (big != NULL) {
      PetscCall(MatCreateVecs(A, big, NULL));
    }

    if (small != NULL) {
      if constexpr (std::is_same_v<MiddleMat, Vec>) {
        PetscCall(VecDuplicate(S, small));
      } else {
        PetscCall(MatCreateVecs(S, small, NULL));
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  Mat A;
  MiddleMat S;
  MiddleMat S_chol;

  Vec z; // work vector
};

struct Coordinate {
  PetscReal x;
  PetscReal y;
};

struct GridOperator {
  enum COLORS : ISColoringValue { RED, BLACK };

  /* Constructs a GridOperator instance for a 2d structured grid of size
   * global_x*global_y and a matrix representing an operator defined on that
   * grid. The parameter mat_assembler must be a function with signature `void
   * mat_assembler(Mat &, DM dm)` that assembles the matrix. Note that
   * the nonzero pattern is alredy set before this function is called, so only
   * the values have to be set (e.g., using PETSc's MatSetValuesStencil).
   */
  template <class MatAssembler>
  GridOperator(PetscInt global_x, PetscInt global_y, Coordinate lower_left,
               Coordinate upper_right, MatAssembler &&mat_assembler)
      : global_x{global_x}, global_y{global_y},
        meshwidth_x{(upper_right.x - lower_left.x) / (global_x - 1)},
        meshwidth_y{(upper_right.y - lower_left.y) / (global_y - 1)} {
    const PetscInt dof_per_node = 1;
    const PetscInt stencil_width = 1;

    PetscFunctionBeginUser;
    PetscCallVoid(DMDACreate2d(PETSC_COMM_WORLD,
                               DM_BOUNDARY_NONE,
                               DM_BOUNDARY_NONE,
                               DMDA_STENCIL_STAR,
                               global_x,
                               global_y,
                               PETSC_DECIDE,
                               PETSC_DECIDE,
                               dof_per_node,
                               stencil_width,
                               NULL,
                               NULL,
                               &dm));
    PetscCallVoid(DMSetUp(dm));
    PetscCallVoid(DMDASetUniformCoordinates(
        dm, lower_left.x, upper_right.x, lower_left.y, upper_right.y, 0, 0));

    // Allocate memory for matrix and initialise non-zero pattern
    PetscCallVoid(DMCreateMatrix(dm, &mat));

    // Call provided assembly functor to fill matrix
    PetscCallVoid(mat_assembler(mat, dm));

    // Create red/black coloring
    PetscCallVoid(setup_coloring());

    MatType type;
    PetscCallVoid(MatGetType(mat, &type));
    if (std::strcmp(type, MATMPIAIJ) == 0) {
      PetscCallVoid(create_rb_scatter());
    }

    PetscFunctionReturnVoid();
  }

  ~GridOperator() {
    PetscFunctionBeginUser;
    PetscCallVoid(MatDestroy(&mat));
    PetscCallVoid(DMDestroy(&dm));
    PetscCallVoid(ISColoringDestroy(&coloring));

    if (sct_vec)
      PetscCallVoid(VecDestroy(&sct_vec));

    // if (scatter_red)
    //   PetscCallVoid(VecScatterDestroy(&scatter_red));

    // if (scatter_black)
    //   PetscCallVoid(VecScatterDestroy(&scatter_black));

    if (scatter)
      PetscCallVoid(VecScatterDestroy(&scatter));

    PetscFunctionReturnVoid();
  }

  void set_lowrank_factor(Mat A, Vec S, Vec S_chol) {
    this->lowrank_update = std::make_unique<LowrankUpdate<Vec>>(A, S, S_chol);
  }

  PetscErrorCode apply(Vec xin, Vec xout) const {
    PetscFunctionBeginUser;

    PetscCall(MatMult(mat, xin, xout));

    if (lowrank_update) {
      Vec tmp; // TODO: Make this a class member to avoid reallocations
      PetscCall(VecDuplicate(xout, &tmp));
      PetscCall(lowrank_update->apply(xin, tmp));
      PetscCall(VecAXPY(xout, 1., tmp));
      PetscCall(VecDestroy(&tmp));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscInt global_x;
  PetscInt global_y;

  PetscReal meshwidth_x;
  PetscReal meshwidth_y;

  DM dm;
  Mat mat;

  // VecScatter scatter_black = nullptr;
  // VecScatter scatter_red = nullptr;
  VecScatter scatter = nullptr;

  ISColoring coloring; // red-black coloring

  std::unique_ptr<LowrankUpdate<Vec>> lowrank_update;

  Vec sct_vec = nullptr;

private:
  PetscErrorCode setup_coloring() {
    PetscFunctionBeginUser;

    const PetscInt ncolors = 2;

    PetscInt start, end;
    PetscCall(MatGetOwnershipRange(mat, &start, &end));

    // Global indices owned by current MPI rank
    std::vector<PetscInt> indices(end - start);
    std::iota(indices.begin(), indices.end(), start);

    // Convert to natural indices
    AO ao;
    PetscCall(DMDAGetAO(dm, &ao));
    PetscCall(AOPetscToApplication(ao, indices.size(), indices.data()));

    std::vector<ISColoringValue> colors(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] % 2 == 0)
        colors[i] = RED;
      else
        colors[i] = BLACK;
    }

    PetscCall(ISColoringCreate(MPI_COMM_WORLD,
                               ncolors,
                               colors.size(),
                               colors.data(),
                               PETSC_COPY_VALUES,
                               &coloring));
    PetscCall(ISColoringViewFromOptions(coloring, NULL, "-rb_coloring_view"));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode create_rb_scatter() {
    PetscFunctionBeginUser;

    Vec global_vec;
    PetscCall(DMGetGlobalVector(dm, &global_vec));

    PetscInt first_row;
    PetscCall(MatGetOwnershipRange(mat, &first_row, NULL));

    Mat B;
    const PetscInt *colmap;
    PetscCall(MatMPIAIJGetSeqAIJ(mat, NULL, &B, &colmap));

    const PetscInt *Bi, *Bj;
    PetscCall(MatSeqAIJGetCSRAndMemType(B, &Bi, &Bj, NULL, NULL));
    PetscInt Brows;
    PetscCall(MatGetSize(B, &Brows, NULL));
    PetscInt Acols;
    PetscCall(MatGetSize(mat, NULL, &Acols));

    /* This array will have non-zero values at indices corresponding to ghost
       vertices. These are identified by looping over the rows of the
       off-diagonal portion B of the given matrix in MPIAIJ format and marking
       columns that contain non-zero entries.

       For example a 5x5 DMDA grid on four processors would lead to an indices
       array (on rank 0) that could look like this:
       indices = {0 0 0 0 0 0 0 0 0 2 0 5 0 8 0 6 7 8 0 0 0 0 0 0 0}.

       The values of the non-zero entries are the rows where B contains non-zero
       columns, using global indexing. These are used below to determine which
       ghost values have to be communicated during red Gibbs sweeps and which
       during black sweeps.
     */
    std::vector<PetscInt> indices(Acols, 0);
    std::size_t nz_cols = 0;
    for (PetscInt row = 0; row < Brows; ++row) {
      for (PetscInt k = Bi[row]; k < Bi[row + 1]; ++k) {
        /* We have to use colmap here since B is compactified, i.e., its
         * non-zero columns are {0, ..., nz_cols} (see MatSetUpMultiply_MPIAIJ).
         */
        if (!indices[colmap[Bj[k]]])
          nz_cols++;
        indices[colmap[Bj[k]]] = 1;
      }
    }

    // Form array of needed columns.
    std::vector<PetscInt> ghost_arr(nz_cols);
    PetscInt cnt = 0;
    for (PetscInt i = 0; i < Acols; ++i)
      if (indices[i])
        ghost_arr[cnt++] = i;

    // PetscCall(PetscIntView(
    //     ghost_arr.size(), ghost_arr.data(), PETSC_VIEWER_STDOUT_WORLD));

    IS from, to;
    PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
                              ghost_arr.size(),
                              ghost_arr.data(),
                              PETSC_COPY_VALUES,
                              &from));
    PetscCall(ISCreateStride(MPI_COMM_WORLD, ghost_arr.size(), 0, 1, &to));

    PetscCall(MatCreateVecs(B, &sct_vec, NULL));
    PetscCall(VecScatterCreate(global_vec, from, sct_vec, to, &scatter));
    PetscCall(
        VecScatterViewFromOptions(scatter, NULL, "-ghost_vecscatter_view"));

    PetscCall(ISDestroy(&from));
    PetscCall(ISDestroy(&to));

    // // Split ghost indices into red and black
    // std::vector<PetscInt> black_ghosts;
    // std::vector<PetscInt> red_ghosts;
    // std::vector<PetscInt> black_target;
    // std::vector<PetscInt> red_target;
    // black_ghosts.reserve((ghost_arr.size() + 1) / 2);
    // red_ghosts.reserve((ghost_arr.size() + 1) / 2);
    // for (std::size_t i = 0; i < nz_cols; ++i) {
    //   const auto ghost = ghost_arr[i];

    //   /* If the "origin index" is red, then the ghost is black. */
    //   if (indices[ghost] % 2 == 0) {
    //     black_ghosts.push_back(ghost);
    //     black_target.push_back(i);
    //   } else {
    //     red_ghosts.push_back(ghost);
    //     red_target.push_back(i);
    //   }
    // }

    // IS from_black, from_red;
    // PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
    //                           black_ghosts.size(),
    //                           black_ghosts.data(),
    //                           PETSC_COPY_VALUES,
    //                           &from_black));
    // PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
    //                           red_ghosts.size(),
    //                           red_ghosts.data(),
    //                           PETSC_COPY_VALUES,
    //                           &from_red));

    // IS to_red, to_black;
    // PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
    //                           black_target.size(),
    //                           black_target.data(),
    //                           PETSC_COPY_VALUES,
    //                           &to_black));
    // PetscCall(ISCreateGeneral(MPI_COMM_WORLD,
    //                           red_target.size(),
    //                           red_target.data(),
    //                           PETSC_COPY_VALUES,
    //                           &to_red));

    // Vec sct_vec;
    // PetscCall(MatCreateVecs(B, &sct_vec, NULL));

    // PetscCall(VecScatterCreate(
    //     global_vec, from_black, sct_vec, to_black, &scatter_black));
    // PetscCall(VecScatterViewFromOptions(
    //     scatter_black, NULL, "-ghost_vecscatter_view"));

    // PetscCall(
    //     VecScatterCreate(global_vec, from_red, sct_vec, to_red,
    //     &scatter_red));
    // PetscCall(
    //     VecScatterViewFromOptions(scatter_red, NULL,
    //     "-ghost_vecscatter_view"));

    // PetscCall(ISDestroy(&to_red));
    // PetscCall(ISDestroy(&to_black));
    // PetscCall(ISDestroy(&from_red));
    // PetscCall(ISDestroy(&from_black));
    PetscCall(DMRestoreGlobalVector(dm, &global_vec));

    PetscFunctionReturn(PETSC_SUCCESS);
  }
};
} // namespace parmgmc
