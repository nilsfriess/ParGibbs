#pragma once

#include <petscconf.h>
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/log.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/linear_operator.hh"

#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscvec.h>

namespace parmgmc {
template <class Engine> class CholeskySampler {
public:
  CholeskySampler(const LinearOperator &linearOperator, Engine &engine)
      : linearOperator{linearOperator}, engine{engine} {
    PetscFunctionBegin;
    PARMGMC_INFO << "Computing Cholesky factorisation...\n";
    Timer timer;

    Mat smat = nullptr;
    if (linearOperator.getMatType() == PetscMatType::MPIAij) {
      PARMGMC_INFO << "\t Converting matrix to right format...";
      PetscCallVoid(MatConvert(linearOperator.getMat(), MATSBAIJ, MAT_INITIAL_MATRIX, &smat));
      PARMGMC_INFO_NP << "done. Took " << timer.elapsed() << " seconds.\n";
      timer.reset();
    } else if (linearOperator.getMatType() == PetscMatType::SEQAij) {
      smat = linearOperator.getMat();
    }

    PetscCallVoid(MatSetOption(smat, MAT_SPD, PETSC_TRUE));

    if (linearOperator.getMatType() == PetscMatType::MPIAij) {
      PetscCallVoid(MatGetFactor(smat, MATSOLVERMKL_CPARDISO, MAT_FACTOR_CHOLESKY, &factor));

      PetscCallVoid(MatMkl_CPardisoSetCntl(factor, 51, 1)); // Use MPI parallel solver
      int mpiSize;
      PetscCallVoid(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
      // TODO: On a cluster, it might be necessary to set these values
      // differently
      PetscCallVoid(MatMkl_CPardisoSetCntl(factor, 52, mpiSize)); // Set numper of MPI ranks
      PetscCallVoid(
          MatMkl_CPardisoSetCntl(factor, 3, 1)); // Set number of OpenMP processes per rank

      // PetscCallVoid(MatMkl_CPardisoSetCntl(factor, 68, 1)); // Message level
      // info

    } else if (linearOperator.getMatType() == PetscMatType::SEQAij) {
      PetscCallVoid(MatGetFactor(smat, MATSOLVERMKL_PARDISO, MAT_FACTOR_CHOLESKY, &factor));
    }

    PetscCallVoid(MatCholeskyFactorSymbolic(factor, smat, nullptr, nullptr));
    PetscCallVoid(MatCholeskyFactorNumeric(factor, smat, nullptr));

    if (linearOperator.getMatType() == PetscMatType::MPIAij)
      PetscCallVoid(MatDestroy(&smat));

    PARMGMC_INFO << "Done. Cholesky factorisation took " << timer.elapsed() << " seconds\n";

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample, const Vec rhs,
                        // n_samples is ignored in the Cholesky sampler since
                        // it always produces an independent sample
                        [[maybe_unused]] std::size_t nSamples = 1) {
    PetscFunctionBeginUser;

    if (v == nullptr) {
      PetscCall(VecDuplicate(rhs, &v));
      PetscCall(VecDuplicate(rhs, &r));
    }

    PetscCall(fillVecRand(r, engine));
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

  CholeskySampler(CholeskySampler &) = delete;
  CholeskySampler &operator=(CholeskySampler &) = delete;

  CholeskySampler(CholeskySampler &&other) noexcept
      : linearOperator{std::move(other.linearOperator)}, engine{other.engine}, factor{other.factor},
        v{other.v}, r{other.r} {
    other.factor = nullptr;
    other.v = nullptr;
    other.r = nullptr;
  }

  CholeskySampler &operator=(CholeskySampler &&) = delete;

private:
  const LinearOperator &linearOperator;
  Engine &engine;

  Mat factor;

  Vec v = nullptr;
  Vec r = nullptr;
};
} // namespace parmgmc
#endif
