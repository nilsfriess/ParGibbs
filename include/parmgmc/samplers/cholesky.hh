#pragma once

#include <petscconf.h>
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/linear_operator.hh"

#include <memory>

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class CholeskySampler {
public:
  CholeskySampler(const std::shared_ptr<LinearOperator> &linear_operator,
                  Engine *engine)
      : linear_operator{linear_operator}, engine{engine} {
    PetscFunctionBegin;
    PARMGMC_INFO << "Computing Cholesky factorisation...\n";
    Timer timer;

    Mat smat;
    if (linear_operator->get_mat_type() == PetscMatType::MPIAIJ) {
      PARMGMC_INFO << "\t Converting matrix to right format...";
      PetscCallVoid(MatConvert(
          linear_operator->get_mat(), MATSBAIJ, MAT_INITIAL_MATRIX, &smat));
      PARMGMC_INFO_NP << "done. Took " << timer.elapsed() << " seconds.\n";
      timer.reset();
    } else if (linear_operator->get_mat_type() == PetscMatType::SEQAIJ) {
      smat = linear_operator->get_mat();
    }

    PetscCallVoid(MatSetOption(smat, MAT_SPD, PETSC_TRUE));

    if (linear_operator->get_mat_type() == PetscMatType::MPIAIJ) {
      PetscCallVoid(MatGetFactor(
          smat, MATSOLVERMKL_CPARDISO, MAT_FACTOR_CHOLESKY, &factor));

      PetscCallVoid(
          MatMkl_CPardisoSetCntl(factor, 51, 1)); // Use MPI parallel solver
      int mpi_size;
      PetscCallVoid(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
      // TODO: On a cluster, it might be necessary to set these values
      // differently
      PetscCallVoid(MatMkl_CPardisoSetCntl(
          factor, 52, mpi_size)); // Set numper of MPI ranks
      PetscCallVoid(MatMkl_CPardisoSetCntl(
          factor, 3, 1)); // Set number of OpenMP processes per rank

      // PetscCallVoid(MatMkl_CPardisoSetCntl(factor, 68, 1)); // Message level
      // info

    } else if (linear_operator->get_mat_type() == PetscMatType::SEQAIJ) {
      PetscCallVoid(MatGetFactor(
          smat, MATSOLVERMKL_PARDISO, MAT_FACTOR_CHOLESKY, &factor));
    }

    PetscCallVoid(MatCholeskyFactorSymbolic(factor, smat, nullptr, nullptr));
    PetscCallVoid(MatCholeskyFactorNumeric(factor, smat, nullptr));

    if (linear_operator->get_mat_type() == PetscMatType::MPIAIJ)
      PetscCallVoid(MatDestroy(&smat));

    PARMGMC_INFO << "Done. Cholesky factorisation took " << timer.elapsed()
                 << " seconds\n";

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, const Vec rhs,
                        // n_samples is ignored in the Cholesky sampler since
                        // it always produces an independent sample
                        [[maybe_unused]] std::size_t n_samples = 1) {
    PetscFunctionBeginUser;

    if (v == nullptr) {
      PetscCall(VecDuplicate(rhs, &v));
      PetscCall(VecDuplicate(rhs, &r));
    }

    PetscCall(fill_vec_rand(r, *engine));
    PetscCall(MatForwardSolve(factor, rhs, v));

    PetscCall(VecAXPY(v, 1., r));

    PetscCall(MatBackwardSolve(factor, v, sample));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~CholeskySampler() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&r));
    PetscCallVoid(VecDestroy(&v));

    PetscCallVoid(MatDestroy(&factor));

    PetscFunctionReturnVoid();
  }

private:
  std::shared_ptr<LinearOperator> linear_operator;
  Engine *engine;

  Mat factor;

  Vec v = nullptr;
  Vec r = nullptr;
};
} // namespace parmgmc
#endif