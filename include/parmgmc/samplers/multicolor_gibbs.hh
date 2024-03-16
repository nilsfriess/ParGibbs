#pragma once

#include "parmgmc/common/coloring.hh"
#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/linear_operator.hh"

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

template <class Engine> class MulticolorGibbsSampler {
public:
  MulticolorGibbsSampler(const std::shared_ptr<LinearOperator> &linear_operator,
                         Engine *engine, PetscReal omega = 1.,
                         GibbsSweepType sweep_type = GibbsSweepType::Forward)
      : linear_operator{linear_operator}, engine{engine}, omega{omega},
        sweep_type{sweep_type} {
    PetscFunctionBeginUser;

    PetscCallVoid(
        MatCreateVecs(linear_operator->get_mat(), &rand_vec, nullptr));
    PetscCallVoid(VecGetLocalSize(rand_vec, &rand_vec_size));

    // Inverse diagonal
    PetscCallVoid(
        MatCreateVecs(linear_operator->get_mat(), &inv_diag_omega, nullptr));
    PetscCallVoid(MatGetDiagonal(linear_operator->get_mat(), inv_diag_omega));
    PetscCallVoid(VecReciprocal(inv_diag_omega));

    // sqrt((2-w)*w) * square root of inverse diagonal
    PetscCallVoid(VecDuplicate(inv_diag_omega, &inv_sqrt_diag_omega));
    PetscCallVoid(VecCopy(inv_diag_omega, inv_sqrt_diag_omega));
    PetscCallVoid(VecSqrtAbs(inv_sqrt_diag_omega));
    PetscCallVoid(
        VecScale(inv_sqrt_diag_omega, std::sqrt((2 - omega) * omega)));

    // Now multiply inv_diag_omega by omega
    PetscCallVoid(VecScale(inv_diag_omega, omega));

    MatType type;
    PetscCallVoid(MatGetType(linear_operator->get_mat(), &type));

    Mat Ad = nullptr;
    if (std::strcmp(type, MATMPIAIJ) == 0) {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(
          linear_operator->get_mat(), &Ad, nullptr, nullptr));
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

    PetscCallVoid(MatSeqAIJGetCSRAndMemType(Ad, &i, &j, &a, nullptr));

    PetscInt rows;
    PetscCallVoid(MatGetSize(Ad, &rows, nullptr));

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

    if (!linear_operator->has_coloring())
      linear_operator->color_matrix();

    PetscFunctionReturnVoid();
  }

  void setSweepType(GibbsSweepType new_type) { sweep_type = new_type; }

  PetscErrorCode setFixedRhs(Vec rhs) {
    PetscFunctionBeginUser;

    if (fixed_rhs == nullptr)
      PetscCall(VecDuplicate(rhs, &fixed_rhs));

    PetscCall(VecCopy(rhs, fixed_rhs));
    PetscCall(VecPointwiseMult(fixed_rhs, fixed_rhs, inv_diag_omega));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode unfixRhs() {
    if (fixed_rhs == nullptr)
      return PETSC_SUCCESS;

    PetscFunctionBeginUser;

    PetscCall(VecDestroy(&fixed_rhs));
    fixed_rhs = nullptr;

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    Vec sampler_rhs;

    if (fixed_rhs)
      sampler_rhs = fixed_rhs;
    else {
      PetscCall(VecDuplicate(rhs, &sampler_rhs));
      PetscCall(VecCopy(rhs, sampler_rhs));
      PetscCall(
          VecPointwiseMult(sampler_rhs, sampler_rhs, inv_diag_omega));
    }

    if (linear_operator->get_mat_type() == PetscMatType::MPIAIJ) {
      for (std::size_t n = 0; n < n_samples; ++n)
        PetscCall(gibbs_rb_mpi(sample, sampler_rhs));
    } else {
      for (std::size_t n = 0; n < n_samples; ++n)
        PetscCall(gibbs_rb_seq(sample, sampler_rhs));
    }

    if (!fixed_rhs)
      PetscCall(VecDestroy(&sampler_rhs));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~MulticolorGibbsSampler() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&rand_vec));
    PetscCallVoid(VecDestroy(&inv_sqrt_diag_omega));
    PetscCallVoid(VecDestroy(&inv_diag_omega));

    PetscCallVoid(unfixRhs());

    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode gibbs_rb_seq(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(PetscHelper::begin_gibbs_event());

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
        linear_operator->get_mat(), &rowptr, &colptr, &matvals, nullptr));

    PetscInt rows;
    PetscCall(MatGetSize(linear_operator->get_mat(), &rows, nullptr));

    const PetscScalar *inv_diag_arr;
    PetscCall(VecGetArrayRead(inv_diag_omega, &inv_diag_arr));

    MatInfo matinfo;
    PetscCall(MatGetInfo(linear_operator->get_mat(), MAT_LOCAL, &matinfo));

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
                        inv_diag_arr[row] * sum;
    };

    if (sweep_type == GibbsSweepType::Forward ||
        sweep_type == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linear_operator->get_coloring()->for_each_color(
          [&](auto /*i*/, const auto &color_indices) {
            for (auto idx : color_indices)
              gibbs_kernel(idx);
          });
    }

    if (sweep_type == GibbsSweepType::Symmetric) {
      PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
      PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
      PetscCall(VecAXPY(rand_vec, 1., rhs));
    }

    if (sweep_type == GibbsSweepType::Backward ||
        sweep_type == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linear_operator->get_coloring()->for_each_color_reverse(
          [&](auto /*i*/, const auto &color_indices) {
            for (auto idx = color_indices.crbegin();
                 idx != color_indices.crend();
                 ++idx)
              gibbs_kernel(*idx);
          });
    }

    PetscCall(VecRestoreArrayRead(inv_diag_omega, &inv_diag_arr));
    PetscCall(VecRestoreArray(sample, &sample_arr));
    PetscCall(VecRestoreArrayRead(rhs, &rand_arr));

    PetscCall(PetscHelper::end_gibbs_event());

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode gibbs_rb_mpi(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(PetscHelper::begin_gibbs_event());

    PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
    PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
    PetscCall(VecAXPY(rand_vec, 1., rhs));

    Mat Ad, Ao;
    PetscCall(
        MatMPIAIJGetSeqAIJ(linear_operator->get_mat(), &Ad, &Ao, nullptr));

    const PetscInt *rowptr, *colptr;
    PetscReal *matvals;
    PetscCall(
        MatSeqAIJGetCSRAndMemType(Ad, &rowptr, &colptr, &matvals, nullptr));

    const PetscInt *B_rowptr, *B_colptr;
    PetscReal *B_matvals;
    PetscCall(MatSeqAIJGetCSRAndMemType(
        Ao, &B_rowptr, &B_colptr, &B_matvals, nullptr));

    PetscInt rows;
    PetscCall(MatGetSize(Ad, &rows, nullptr));

    PetscReal *sample_arr;
    const PetscReal *rand_arr, *inv_diag_arr, *ghost_arr;

    PetscCall(VecGetArrayRead(inv_diag_omega, &inv_diag_arr));
    PetscCall(VecGetArrayRead(rand_vec, &rand_arr));

    MatInfo matinfo;
    PetscCall(MatGetInfo(linear_operator->get_mat(), MAT_LOCAL, &matinfo));

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

      for (PetscInt k = 0; k < B_rowptr[row + 1] - B_rowptr[row]; ++k)
        sum -= B_matvals[B_rowptr[row] + k] * ghost_arr[k];

      // Update sample
      sample_arr[row] = (1 - omega) * sample_arr[row] + rand_arr[row] +
                        inv_diag_arr[row] * sum;
    };

    PetscInt first_row, last_row;
    PetscCall(MatGetOwnershipRange(
        linear_operator->get_mat(), &first_row, &last_row));

    if (sweep_type == GibbsSweepType::Forward ||
        sweep_type == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linear_operator->get_coloring()->for_each_color(
          [&](auto i, const auto &color_indices) {
            PetscFunctionBeginUser;

            auto scatter = linear_operator->get_coloring()->get_scatter(i);
            auto ghostvec = linear_operator->get_coloring()->get_ghost_vec(i);

            PetscCall(VecScatterBegin(
                scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));
            PetscCall(VecScatterEnd(
                scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));

            PetscCall(VecGetArrayRead(ghostvec, &ghost_arr));
            PetscCall(VecGetArray(sample, &sample_arr));

            for (auto idx : color_indices)
              gibbs_kernel(idx);

            PetscCall(VecRestoreArray(sample, &sample_arr));
            PetscCall(VecRestoreArrayRead(ghostvec, &ghost_arr));

            PetscFunctionReturn(PETSC_SUCCESS);
          });
    }

    if (sweep_type == GibbsSweepType::Symmetric) {
      PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
      PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
      PetscCall(VecAXPY(rand_vec, 1., rhs));
    }

    if (sweep_type == GibbsSweepType::Backward ||
        sweep_type == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linear_operator->get_coloring()->for_each_color_reverse(
          [&](auto i, const auto &color_indices) {
            PetscFunctionBeginUser;

            auto scatter = linear_operator->get_coloring()->get_scatter(i);
            auto ghostvec = linear_operator->get_coloring()->get_ghost_vec(i);

            PetscCall(VecScatterBegin(
                scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));
            PetscCall(VecScatterEnd(
                scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));

            PetscCall(VecGetArrayRead(ghostvec, &ghost_arr));
            PetscCall(VecGetArray(sample, &sample_arr));

            for (auto idx = color_indices.crbegin();
                 idx != color_indices.crend();
                 ++idx)
              gibbs_kernel(*idx);

            PetscCall(VecRestoreArray(sample, &sample_arr));
            PetscCall(VecRestoreArrayRead(ghostvec, &ghost_arr));

            PetscFunctionReturn(PETSC_SUCCESS);
          });
    }

    PetscCall(VecRestoreArrayRead(rand_vec, &rand_arr));
    PetscCall(VecRestoreArrayRead(inv_diag_omega, &inv_diag_arr));

    PetscCall(PetscHelper::end_gibbs_event());

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::shared_ptr<LinearOperator> linear_operator;

  Engine *engine;
  PetscReal omega;

  Vec inv_sqrt_diag_omega;
  Vec inv_diag_omega;

  Vec rand_vec;
  PetscInt rand_vec_size;

  Vec fixed_rhs = nullptr;

  GibbsSweepType sweep_type;

  /// Only used in parallel execution
  std::vector<PetscInt> diag_ptrs; // Indices of the diagonal entries
};

} // namespace parmgmc
