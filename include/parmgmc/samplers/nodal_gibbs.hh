#pragma once

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/types.hh"
#include "parmgmc/linear_operator.hh"

#include <memory>
#include <petscerror.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class NodalGibbsSampler {
public:
  NodalGibbsSampler(const std::shared_ptr<LinearOperator> &linear_operator,
                    Engine *engine)
      : linear_operator{linear_operator}, engine{engine} {
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

  PetscErrorCode sample(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    BotMidTopPartition partition;
    PetscCall(make_botmidtop_partition(linear_operator->get_mat(), partition));

    PetscLogEvent gibbs_event;
    PetscCall(PetscHelper::get_gibbs_event(&gibbs_event));
    PetscCall(
        PetscLogEventBegin(gibbs_event, nullptr, nullptr, nullptr, nullptr));

    PetscCall(fill_vec_rand(rand_vec, rand_vec_size, *engine));
    PetscCall(VecPointwiseMult(rand_vec, rand_vec, inv_sqrt_diag_omega));
    PetscCall(VecAXPY(rand_vec, 1., rhs));

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

      for (PetscInt k = B_rowptr[row]; k < B_rowptr[row + 1]; ++k)
        sum -= B_matvals[k] * ghost_arr[B_colptr[k]];

      // Update sample
      sample_arr[row] = rand_arr[row] + inv_diag_arr[row] * sum;
    };

    // 1. Send boundary values to higher processors
    PetscCall(VecScatterBegin(partition.botscatter,
                              sample,
                              ghost_vec,
                              INSERT_VALUES,
                              SCATTER_FORWARD));

    // 2. Receive boundary values from lower processors
    PetscCall(VecScatterEnd(partition.botscatter,
                            sample,
                            ghost_vec,
                            INSERT_VALUES,
                            SCATTER_FORWARD));

    // 3. Run Gibbs kernel on TOP nodes
    for (auto i : partition.top)
      gibbs_kernel(i);

    // 4. Send boundary values to lower processors
    PetscCall(VecScatterBegin(partition.topscatter,
                              sample,
                              ghost_vec,
                              INSERT_VALUES,
                              SCATTER_FORWARD));

    // 5. Run Gibbs kernel on INT1 nodes
    for (auto i : partition.interior1)
      gibbs_kernel(i);

    // 6. Receive boundary values from higher processors
    PetscCall(VecScatterEnd(partition.topscatter,
                            sample,
                            ghost_vec,
                            INSERT_VALUES,
                            SCATTER_FORWARD));

    // 7. Handle mid nodes

    // 8. Run Gibbs on INT2 nodes
    for (auto i : partition.interior2)
      gibbs_kernel(i);

    // 9. Receive boundary values from remaining ghost nodes (sent in step 7)

    // 10. Run Gibbs on BOT nodes
    for (auto i : partition.bot)
      gibbs_kernel(i);

    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  std::shared_ptr<LinearOperator> linear_operator;

  Engine *engine;

  Vec inv_sqrt_diag_omega;
  Vec inv_diag;

  Vec rand_vec;
  PetscInt rand_vec_size;

  Vec ghost_vec = nullptr;
  std::vector<PetscInt> diag_ptrs; // Indices of the diagonal entries
};
} // namespace parmgmc
