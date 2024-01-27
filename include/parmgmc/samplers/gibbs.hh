#pragma once

#include "parmgmc/common/helpers.hh"
#include "parmgmc/linear_operator.hh"

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
  GibbsSampler(std::shared_ptr<LinearOperator> linear_operator, Engine *engine,
               PetscReal omega = 1.,
               GibbsSweepType sweep_type = GibbsSweepType::Forward)
      : linear_operator{linear_operator}, engine{engine}, omega{omega},
        sweep_type{sweep_type} {
    PetscFunctionBeginUser;

    PetscCallVoid(MatCreateVecs(linear_operator->get_mat(), &rand_vec, NULL));
    PetscCallVoid(VecGetLocalSize(rand_vec, &rand_vec_size));

    // Inverse diagonal
    PetscCallVoid(MatCreateVecs(linear_operator->get_mat(), &inv_diag, NULL));
    PetscCallVoid(MatGetDiagonal(linear_operator->get_mat(), inv_diag));
    PetscCallVoid(VecReciprocal(inv_diag));

    // sqrt((2-w)*w) * square root of inverse diagonal
    PetscCallVoid(VecDuplicate(inv_diag, &inv_sqrt_diag_omega));
    PetscCallVoid(VecCopy(inv_diag, inv_sqrt_diag_omega));
    PetscCallVoid(VecSqrtAbs(inv_sqrt_diag_omega));
    PetscCallVoid(
        VecScale(inv_sqrt_diag_omega, std::sqrt((2 - omega) * omega)));

    // const PetscScalar *val;
    // PetscCallVoid(VecGetArrayRead(inv_sqrt_diag_omega, &val));
    // PetscCallVoid(PetscPrintf(MPI_COMM_WORLD, "id = %f\n", val[0]));
    // PetscCallVoid(VecRestoreArrayRead(inv_sqrt_diag_omega, &val));

    MatType type;
    PetscCallVoid(MatGetType(linear_operator->get_mat(), &type));

    Mat Ad = nullptr;
    if (std::strcmp(type, MATMPIAIJ) == 0) {
      PetscCallVoid(
          MatMPIAIJGetSeqAIJ(linear_operator->get_mat(), &Ad, NULL, NULL));
    } else if (std::strcmp(type, MATSEQAIJ) == 0) {
      Ad = linear_operator->get_mat();
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
    PetscCall(MatGetType(linear_operator->get_mat(), &type));

    if (std::strcmp(type, MATMPIAIJ) == 0) {
      if (!ghost_vec) {
        Mat Ao;
        PetscCall(
            MatMPIAIJGetSeqAIJ(linear_operator->get_mat(), NULL, &Ao, NULL));
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
  // private:
  // TODO: Add FLOP logging
  PetscErrorCode gibbs_rb_seq(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
    PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
    PetscCall(VecAXPY(rand_vec, 1., rhs));

    PetscReal *sample_arr;
    const PetscReal *rand_arr;

    PetscCall(VecGetArray(sample, &sample_arr));
    PetscCall(VecGetArrayRead(rand_vec, &rand_arr));

    const PetscInt *rowptr, *colptr;
    PetscReal *matvals;

    PetscCall(MatSeqAIJGetCSRAndMemType(
        linear_operator->get_mat(), &rowptr, &colptr, &matvals, NULL));

    PetscInt rows;
    PetscCall(MatGetSize(linear_operator->get_mat(), &rows, NULL));

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

    IS *is_colorings = nullptr;
    PetscInt n_colors;
    if (linear_operator->has_coloring()) {
      PetscCall(ISColoringGetIS(linear_operator->get_coloring(),
                                PETSC_USE_POINTER,
                                &n_colors,
                                &is_colorings));
    }

    if (sweep_type == GibbsSweepType::Forward ||
        sweep_type == GibbsSweepType::Symmetric) {
      if (linear_operator->has_coloring()) {
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
      } else {
        for (PetscInt i = 0; i < rows; ++i) {
          gibbs_kernel(i);
        }
      }
    }

    if (sweep_type == GibbsSweepType::Backward ||
        sweep_type == GibbsSweepType::Symmetric) {
      if (linear_operator->has_coloring()) {
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
      } else {
        for (PetscInt i = rows - 1; i >= 0; --i) {
          gibbs_kernel(i);
        }
      }
    }

    if (linear_operator->has_coloring())
      PetscCall(ISColoringRestoreIS(
          linear_operator->get_coloring(), PETSC_USE_POINTER, &is_colorings));

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
    // PetscCall(VecCopy(rhs, rand_vec));
    // PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_diag));

    Mat Ad, Ao;
    PetscCall(MatMPIAIJGetSeqAIJ(linear_operator->get_mat(), &Ad, &Ao, NULL));

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

    PetscInt first_row, last_row;
    PetscCall(MatGetOwnershipRange(
        linear_operator->get_mat(), &first_row, &last_row));

    if (linear_operator->has_coloring()) {
      IS *is_colorings;
      PetscInt n_colors;
      PetscCall(ISColoringGetIS(linear_operator->get_coloring(),
                                PETSC_USE_POINTER,
                                &n_colors,
                                &is_colorings));

      if (sweep_type == GibbsSweepType::Forward ||
          sweep_type == GibbsSweepType::Symmetric) {
        for (PetscInt color = 0; color < n_colors; ++color) {
          PetscCall(VecZeroEntries(ghost_vec));
          PetscCall(VecScatterBegin(linear_operator->get_scatter(),
                                    sample,
                                    ghost_vec,
                                    INSERT_VALUES,
                                    SCATTER_FORWARD));

          PetscInt n_indices;
          PetscCall(ISGetLocalSize(is_colorings[color], &n_indices));

          const PetscInt *indices;
          PetscCall(ISGetIndices(is_colorings[color], &indices));

          PetscCall(VecScatterEnd(linear_operator->get_scatter(),
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
          PetscCall(VecScatterBegin(linear_operator->get_scatter(),
                                    sample,
                                    ghost_vec,
                                    INSERT_VALUES,
                                    SCATTER_FORWARD));

          PetscInt n_indices;
          PetscCall(ISGetLocalSize(is_colorings[color], &n_indices));

          const PetscInt *indices;
          PetscCall(ISGetIndices(is_colorings[color], &indices));

          PetscCall(VecScatterEnd(linear_operator->get_scatter(),
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

        PetscCall(ISColoringRestoreIS(
            linear_operator->get_coloring(), PETSC_USE_POINTER, &is_colorings));
      }
    } else {
      assert(false && "Not implemented");
    } // { // coloring_type == ColoringType::None
    //   PetscCall(VecZeroEntries(ghost_vec));
    //   PetscCall(VecScatterBegin(grid_operator->scatter,
    //                             sample,
    //                             ghost_vec,
    //                             INSERT_VALUES,
    //                             SCATTER_FORWARD));
    //   PetscCall(VecScatterEnd(grid_operator->scatter,
    //                           sample,
    //                           ghost_vec,
    //                           INSERT_VALUES,
    //                           SCATTER_FORWARD));

    //   PetscCall(VecGetArray(sample, &sample_arr));
    //   PetscCall(VecGetArrayRead(ghost_vec, &ghost_arr));

    //   if (sweep_type == GibbsSweepType::Forward ||
    //       sweep_type == GibbsSweepType::Symmetric) {
    //     for (PetscInt i = 0; i < last_row - first_row; ++i)
    //       gibbs_kernel(i);
    //   }

    //   if (sweep_type == GibbsSweepType::Backward ||
    //       sweep_type == GibbsSweepType::Symmetric) {
    //     for (PetscInt i = last_row - first_row - 1; i >= 0; i--)
    //       gibbs_kernel(i);
    //   }

    //   PetscCall(VecRestoreArrayRead(ghost_vec, &ghost_arr));
    //   PetscCall(VecRestoreArray(sample, &sample_arr));
    // }

    PetscCall(VecRestoreArrayRead(rand_vec, &rand_arr));
    PetscCall(VecRestoreArrayRead(inv_diag, &inv_diag_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::shared_ptr<LinearOperator> linear_operator;

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

} // namespace parmgmc
