#pragma once

#include "parmgmc/common/helpers.hh"
#include "parmgmc/grid/grid_operator.hh"

#include <cassert>
#include <cstring>
#include <memory>

#include <mpi.h>
#include <petscdm.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscmat.h>
#include <petscpctypes.h>
#include <petscsftypes.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

namespace parmgmc {
enum class GibbsSweepType { Forward, Backward, Symmetric };

template <class Engine> class GibbsSampler {
public:
  GibbsSampler(std::shared_ptr<GridOperator> grid_operator, Engine *engine,
               PetscReal omega = 1.,
               GibbsSweepType sweep_type = GibbsSweepType::Forward)
      : grid_operator{grid_operator}, engine{engine}, omega{omega},
        sweep_type{sweep_type} {
    PetscFunctionBeginUser;

    PetscCallVoid(MatCreateVecs(grid_operator->mat, &rand_vec, NULL));
    PetscCallVoid(VecGetLocalSize(rand_vec, &rand_vec_size));

    // Inverse diagonal
    PetscCallVoid(MatCreateVecs(grid_operator->mat, &inv_diag, NULL));
    PetscCallVoid(MatGetDiagonal(grid_operator->mat, inv_diag));
    PetscCallVoid(VecReciprocal(inv_diag));

    // sqrt((2-w)*w) * square root of inverse diagonal
    PetscCallVoid(VecDuplicate(inv_diag, &inv_sqrt_diag_omega));
    PetscCallVoid(VecCopy(inv_diag, inv_sqrt_diag_omega));
    PetscCallVoid(VecSqrtAbs(inv_sqrt_diag_omega));
    PetscCallVoid(
        VecScale(inv_sqrt_diag_omega, std::sqrt((2 - omega) * omega)));

    MatType type;
    PetscCallVoid(MatGetType(grid_operator->mat, &type));

    Mat Ad = nullptr;
    if (std::strcmp(type, MATMPIAIJ) == 0) {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(grid_operator->mat, &Ad, NULL, NULL));
    } else if (std::strcmp(type, MATSEQAIJ) == 0) {
      Ad = grid_operator->mat;
    } else {
      PetscCheckAbort(false,
                      MPI_COMM_WORLD,
                      PETSC_ERR_SUP,
                      "Only MATMPIAIJ and MATSEQAIJ types are supported");
    }

    const PetscInt *i, *j;
    PetscReal *a;

    PetscCallVoid(MatSeqAIJGetCSRAndMemType(Ad, &i, &j, &a, NULL));

    PetscInt rows;
    PetscCallVoid(MatGetSize(Ad, &rows, NULL));

    diag_ptrs.reserve(rows);
    for (PetscInt row = 0; row < rows; ++row) {
      const auto row_start = i[row];
      const auto row_end = i[row + 1];

      for (PetscInt k = row_start; k < row_end; ++k) {
        const auto col = j[k];
        if (col == row)
          diag_ptrs.push_back(k);
      }
    }
    PetscCheckAbort(diag_ptrs.size() == (std::size_t)rows,
                    MPI_COMM_WORLD,
                    PETSC_ERR_SUP,
                    "Diagonal elements of precision matrix cannot be zero");

    PetscFunctionReturnVoid();
  }

  void setSweepType(GibbsSweepType new_type) { sweep_type = new_type; }

  PetscErrorCode sample(Vec sample, const Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    // PetscCall(VecAXPY(rand_vec, 1., rhs));

    MatType type;
    PetscCall(MatGetType(grid_operator->mat, &type));

    if (std::strcmp(type, MATMPIAIJ) == 0) {
      if (!ghost_vec) {
        Mat Ao;
        PetscCall(MatMPIAIJGetSeqAIJ(grid_operator->mat, NULL, &Ao, NULL));
        PetscCall(MatCreateVecs(Ao, &ghost_vec, NULL));
      }

      for (std::size_t n = 0; n < n_samples; ++n)
        PetscCall(gibbs_rb_mpi(sample, rhs));
    } else if (std::strcmp(type, MATSEQAIJ) == 0) {
      for (std::size_t n = 0; n < n_samples; ++n)
        PetscCall(gibbs_rb_seq(sample, rhs));
    } else {
      PetscCheck(false,
                 MPI_COMM_WORLD,
                 PETSC_ERR_SUP,
                 "Only MATMPIAIJ and MATSEQAIJ types are supported");
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~GibbsSampler() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&rand_vec));
    PetscCallVoid(VecDestroy(&inv_sqrt_diag_omega));
    PetscCallVoid(VecDestroy(&inv_diag));
    PetscCallVoid(VecDestroy(&ghost_vec));

    PetscFunctionReturnVoid();
  }

private:
  // TODO: Add FLOP logging
  PetscErrorCode gibbs_rb_seq(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
    PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
    PetscCall(VecAXPY(rand_vec, 1., rhs));

    // PetscCall(MatSOR(grid_operator->mat,
    //                  rand_vec,
    //                  omega,
    //                  SOR_SYMMETRIC_SWEEP,
    //                  0,
    //                  1,
    //                  1,
    //                  sample));

    PetscReal *sample_arr;
    const PetscReal *rand_arr;

    PetscCall(VecGetArray(sample, &sample_arr));
    PetscCall(VecGetArrayRead(rand_vec, &rand_arr));

    const PetscInt *rowptr, *colptr;
    PetscReal *matvals;

    PetscCall(MatSeqAIJGetCSRAndMemType(
        grid_operator->mat, &rowptr, &colptr, &matvals, NULL));

    PetscInt rows;
    PetscCall(MatGetSize(grid_operator->mat, &rows, NULL));

    const PetscScalar *inv_diag_arr;
    PetscCall(VecGetArrayRead(inv_diag, &inv_diag_arr));

    const auto gibbs_kernel = [&](PetscInt row) {
      const auto row_start = rowptr[row];
      const auto row_end = rowptr[row + 1];

      PetscReal sum = 0.;

      // Lower triangular part
      const auto n_below = diag_ptrs[row] - row_start;
      for (PetscInt k = 0; k < n_below; ++k)
        sum -= matvals[row_start + k] * sample_arr[colptr[row_start + k]];

      // Upper triangular part
      const auto n_above = row_end - diag_ptrs[row] - 1;
      for (PetscInt k = 0; k < n_above; ++k)
        sum -= matvals[diag_ptrs[row] + 1 + k] *
               sample_arr[colptr[diag_ptrs[row] + 1 + k]];

      // Update sample
      sample_arr[row] = (1 - omega) * sample_arr[row] + rand_arr[row] +
                        omega * inv_diag_arr[row] * sum;
    };

    IS *is_colorings;
    PetscInt n_colors;
    PetscCall(ISColoringGetIS(
        grid_operator->coloring, PETSC_USE_POINTER, &n_colors, &is_colorings));

    if (sweep_type == GibbsSweepType::Forward ||
        sweep_type == GibbsSweepType::Symmetric) {
      for (PetscInt color = 0; color < n_colors; ++color) {
        PetscInt n_indices;
        PetscCall(ISGetLocalSize(is_colorings[color], &n_indices));

        const PetscInt *indices;
        PetscCall(ISGetIndices(is_colorings[color], &indices));

        for (PetscInt i = 0; i < n_indices; ++i) {
          gibbs_kernel(indices[i]);
        }

        PetscCall(ISRestoreIndices(is_colorings[color], &indices));
      }
    }

    if (sweep_type == GibbsSweepType::Backward ||
        sweep_type == GibbsSweepType::Symmetric) {
      for (PetscInt color = n_colors - 1; color >= 0; color--) {
        PetscInt n_indices;
        PetscCall(ISGetLocalSize(is_colorings[color], &n_indices));

        const PetscInt *indices;
        PetscCall(ISGetIndices(is_colorings[color], &indices));

        for (PetscInt i = n_indices - 1; i >= 0; i--) {
          gibbs_kernel(indices[i]);
        }

        PetscCall(ISRestoreIndices(is_colorings[color], &indices));
      }
    }

    PetscCall(ISColoringRestoreIS(
        grid_operator->coloring, PETSC_USE_POINTER, &is_colorings));

    PetscCall(VecRestoreArrayRead(inv_diag, &inv_diag_arr));
    PetscCall(VecRestoreArray(sample, &sample_arr));
    PetscCall(VecRestoreArrayRead(rhs, &rand_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode gibbs_rb_mpi(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
    PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
    PetscCall(VecAXPY(rand_vec, 1., rhs));

    Mat Ad, Ao;
    PetscCall(MatMPIAIJGetSeqAIJ(grid_operator->mat, &Ad, &Ao, NULL));

    const PetscInt *rowptr, *colptr;
    PetscReal *matvals;
    PetscCall(MatSeqAIJGetCSRAndMemType(Ad, &rowptr, &colptr, &matvals, NULL));

    const PetscInt *B_rowptr, *B_colptr;
    PetscReal *B_matvals;
    PetscCall(
        MatSeqAIJGetCSRAndMemType(Ao, &B_rowptr, &B_colptr, &B_matvals, NULL));

    PetscInt rows;
    PetscCall(MatGetSize(Ad, &rows, NULL));

    PetscReal *sample_arr;
    const PetscReal *rand_arr, *inv_diag_arr, *ghost_arr;

    PetscCall(VecGetArrayRead(inv_diag, &inv_diag_arr));
    PetscCall(VecGetArrayRead(rand_vec, &rand_arr));

    const auto gibbs_kernel = [&](PetscInt row) {
      const auto row_start = rowptr[row];
      const auto row_end = rowptr[row + 1];
      const auto row_diag = diag_ptrs[row];

      PetscReal sum = 0.;

      // Lower triangular part
      for (PetscInt k = row_start; k < row_diag; ++k)
        sum -= matvals[k] * sample_arr[colptr[k]];

      // Upper triangular part
      for (PetscInt k = row_diag + 1; k < row_end; ++k)
        sum -= matvals[k] * sample_arr[colptr[k]];

      for (PetscInt k = B_rowptr[row]; k < B_rowptr[row + 1]; ++k)
        sum -= B_matvals[k] * ghost_arr[B_colptr[k]];

      // Update sample
      sample_arr[row] = (1 - omega) * sample_arr[row] + rand_arr[row] +
                        omega * inv_diag_arr[row] * sum;
    };

    PetscInt first_row;
    PetscCall(MatGetOwnershipRange(grid_operator->mat, &first_row, NULL));

    IS *is_colorings;
    PetscInt n_colors;
    PetscCall(ISColoringGetIS(
        grid_operator->coloring, PETSC_USE_POINTER, &n_colors, &is_colorings));

    if (sweep_type == GibbsSweepType::Forward ||
        sweep_type == GibbsSweepType::Symmetric) {
      for (PetscInt color = 0; color < n_colors; ++color) {
        PetscCall(VecZeroEntries(ghost_vec));
        PetscCall(VecScatterBegin(grid_operator->scatter,
                                  sample,
                                  ghost_vec,
                                  INSERT_VALUES,
                                  SCATTER_FORWARD));

        PetscInt n_indices;
        PetscCall(ISGetLocalSize(is_colorings[color], &n_indices));

        const PetscInt *indices;
        PetscCall(ISGetIndices(is_colorings[color], &indices));

        PetscCall(VecScatterEnd(grid_operator->scatter,
                                sample,
                                ghost_vec,
                                INSERT_VALUES,
                                SCATTER_FORWARD));

        PetscCall(VecGetArray(sample, &sample_arr));
        PetscCall(VecGetArrayRead(ghost_vec, &ghost_arr));

        // PetscCall(PetscIntView(n_indices, indices,
        // PETSC_VIEWER_STDOUT_WORLD));

        for (PetscInt i = 0; i < n_indices; ++i) {
          gibbs_kernel(indices[i]);
        }

        PetscCall(VecRestoreArrayRead(ghost_vec, &ghost_arr));
        PetscCall(VecRestoreArray(sample, &sample_arr));
        PetscCall(ISRestoreIndices(is_colorings[color], &indices));

        // PetscCall(VecView(sample, PETSC_VIEWER_STDOUT_WORLD));
      }
    }

    if (sweep_type == GibbsSweepType::Backward ||
        sweep_type == GibbsSweepType::Symmetric) {
      for (PetscInt color = n_colors - 1; color >= 0; color--) {
        PetscCall(VecZeroEntries(ghost_vec));
        PetscCall(VecScatterBegin(grid_operator->scatter,
                                  sample,
                                  ghost_vec,
                                  INSERT_VALUES,
                                  SCATTER_FORWARD));

        PetscInt n_indices;
        PetscCall(ISGetLocalSize(is_colorings[color], &n_indices));

        const PetscInt *indices;
        PetscCall(ISGetIndices(is_colorings[color], &indices));

        PetscCall(VecScatterEnd(grid_operator->scatter,
                                sample,
                                ghost_vec,
                                INSERT_VALUES,
                                SCATTER_FORWARD));

        PetscCall(VecGetArray(sample, &sample_arr));
        PetscCall(VecGetArrayRead(ghost_vec, &ghost_arr));

        for (PetscInt i = n_indices - 1; i >= 0; i--) {
          gibbs_kernel(indices[i]);
        }

        PetscCall(VecRestoreArrayRead(ghost_vec, &ghost_arr));
        PetscCall(VecRestoreArray(sample, &sample_arr));
        PetscCall(ISRestoreIndices(is_colorings[color], &indices));
      }
    }

    PetscCall(ISColoringRestoreIS(
        grid_operator->coloring, PETSC_USE_POINTER, &is_colorings));

    PetscCall(VecRestoreArrayRead(rand_vec, &rand_arr));
    PetscCall(VecRestoreArrayRead(inv_diag, &inv_diag_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::shared_ptr<GridOperator> grid_operator;
  Engine *engine;
  PetscReal omega;

  Vec inv_sqrt_diag_omega;
  Vec inv_diag;

  Vec rand_vec;
  PetscInt rand_vec_size;

  // PetscLogEvent GIBBS_RB;
  GibbsSweepType sweep_type;

  /// Only used in parallel execution
  Vec ghost_vec = nullptr;
  std::vector<PetscInt> diag_ptrs; // Indices of the diagonal entries
};

template <class Engine>
inline PetscErrorCode
PCShellCallback_Gibbs(PC pc, Vec b, Vec x, Vec r, PetscReal rtol,
                      PetscReal abstol, PetscReal dtol, PetscInt maxits,
                      PetscBool zeroinitialguess, PetscInt *its,
                      PCRichardsonConvergedReason *reason) {
  /* We ignore all the provided tolerances since this is only supposed to be
   * used within MGMC
   */
  (void)rtol;
  (void)abstol;
  (void)dtol;
  (void)maxits;
  (void)r;

  // Assume for now that x is not zero
  (void)zeroinitialguess;

  // Always return one iteration
  *its = 1;
  *reason = PCRICHARDSON_CONVERGED_ITS;

  PetscFunctionBeginUser;

  GibbsSampler<Engine> *sampler;
  PetscCall(PCShellGetContext(pc, &sampler));

  PetscCall(sampler->sample(x, b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace parmgmc
