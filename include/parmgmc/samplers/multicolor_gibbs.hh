#pragma once

#include "parmgmc/common/coloring.hh"
#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/linear_operator.hh"

#include <cstring>

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

template <class Engine>
class MulticolorGibbsSampler {
public:
  MulticolorGibbsSampler(LinearOperator &linearOperator, Engine &engine, PetscReal omega = 1.,
                         GibbsSweepType sweepType = GibbsSweepType::Forward)
      : linearOperator{linearOperator}, engine{engine}, omega{omega}, sweepType{sweepType} {
    PetscFunctionBeginUser;

    PetscCallVoid(MatCreateVecs(linearOperator.getMat(), &randVec, nullptr));
    PetscCallVoid(VecGetLocalSize(randVec, &randVecSize));

    PetscCallVoid(MatCreateVecs(linearOperator.getMat(), &sqrtDiagOmega, nullptr));
    PetscCallVoid(MatGetDiagonal(linearOperator.getMat(), sqrtDiagOmega));
    PetscCallVoid(VecSqrtAbs(sqrtDiagOmega));
    PetscCallVoid(VecScale(sqrtDiagOmega, std::sqrt((2 - omega) / omega)));

    // Inverse diagonal
    PetscCallVoid(MatCreateVecs(linearOperator.getMat(), &invDiagOmega, nullptr));
    PetscCallVoid(MatGetDiagonal(linearOperator.getMat(), invDiagOmega));
    PetscCallVoid(VecReciprocal(invDiagOmega));
    PetscCallVoid(VecScale(invDiagOmega, omega));

    MatType type;
    PetscCallVoid(MatGetType(linearOperator.getMat(), &type));

    Mat ad = nullptr;
    if (std::strcmp(type, MATMPIAIJ) == 0) {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(linearOperator.getMat(), &ad, nullptr, nullptr));
    } else if (std::strcmp(type, MATSEQAIJ) == 0) {
      ad = linearOperator.getMat();
    } else {
      PetscCheckAbort(false, MPI_COMM_WORLD, PETSC_ERR_SUP,
                      "Only MATMPIAIJ and MATSEQAIJ types are supported");
    }

    const PetscInt *i, *j;
    PetscReal *a;

    PetscCallVoid(MatSeqAIJGetCSRAndMemType(ad, &i, &j, &a, nullptr));

    PetscInt rows;
    PetscCallVoid(MatGetSize(ad, &rows, nullptr));

    diagPtrs.reserve(rows);
    for (PetscInt row = 0; row < rows; ++row) {
      const auto rowStart = i[row];
      const auto rowEnd = i[row + 1];

      for (PetscInt k = rowStart; k < rowEnd; ++k) {
        const auto col = j[k];
        if (col == row)
          diagPtrs.push_back(k);
      }
    }
    PetscCheckAbort(diagPtrs.size() == (std::size_t)rows, MPI_COMM_WORLD, PETSC_ERR_SUP,
                    "Diagonal elements of precision matrix cannot be zero");

    if (!linearOperator.hasColoring())
      linearOperator.colorMatrix();

    PetscFunctionReturnVoid();
  }

  // MulticolorGibbsSampler(MulticolorGibbsSampler &&other) noexcept
  //     : linearOperator{std::move(other.linearOperator)}, engine{other.engine},
  //     omega{other.omega},
  //       sqrtDiagOmega{other.sqrtDiagOmega}, invDiagOmega{other.invDiagOmega},
  //       randVec{other.randVec}, randVecSize{other.randVecSize}, sweepType{other.sweepType},
  //       diagPtrs{std::move(other.diagPtrs)} {
  //   other.sqrtDiagOmega = nullptr;
  //   other.invDiagOmega = nullptr;
  //   other.randVec = nullptr;
  // }

  void setSweepType(GibbsSweepType newType) { sweepType = newType; }

  PetscErrorCode sample(Vec sample, Vec rhs, std::size_t nSamples = 1) {
    PetscFunctionBeginUser;

    if (linearOperator.getMatType() == PetscMatType::MPIAij) {
      for (std::size_t n = 0; n < nSamples; ++n)
        PetscCall(gibbsRbMpi(sample, rhs));
    } else {
      for (std::size_t n = 0; n < nSamples; ++n)
        PetscCall(gibbsRbSeq(sample, rhs));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~MulticolorGibbsSampler() {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDestroy(&randVec));
    PetscCallVoid(VecDestroy(&sqrtDiagOmega));
    PetscCallVoid(VecDestroy(&invDiagOmega));

    PetscFunctionReturnVoid();
  }

private:
  PetscErrorCode gibbsRbSeq(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(PetscHelper::beginGibbsEvent());

    PetscCall(fillVecRand(randVec, randVecSize, engine));
    PetscCall(VecPointwiseMult(randVec, randVec, sqrtDiagOmega));
    PetscCall(VecAXPY(randVec, 1., rhs));
    PetscCall(VecPointwiseMult(randVec, randVec, invDiagOmega));

    PetscReal *sampleArr;
    const PetscReal *randArr;

    PetscCall(VecGetArray(sample, &sampleArr));
    PetscCall(VecGetArrayRead(randVec, &randArr));

    const PetscInt *rowptr, *colptr;
    PetscReal *matvals;
    PetscCall(
        MatSeqAIJGetCSRAndMemType(linearOperator.getMat(), &rowptr, &colptr, &matvals, nullptr));

    PetscInt rows;
    PetscCall(MatGetSize(linearOperator.getMat(), &rows, nullptr));

    const PetscScalar *invDiagArr;
    PetscCall(VecGetArrayRead(invDiagOmega, &invDiagArr));

    MatInfo matinfo;
    PetscCall(MatGetInfo(linearOperator.getMat(), MAT_LOCAL, &matinfo));

    const auto gibbsKernel = [&](PetscInt row) {
      const auto rowStart = rowptr[row];
      const auto rowEnd = rowptr[row + 1];

      PetscReal sum = 0.;

      // Lower triangular part
      const auto nBelow = diagPtrs[row] - rowStart;
      for (PetscInt k = 0; k < nBelow; ++k)
        sum -= matvals[rowStart + k] * sampleArr[colptr[rowStart + k]];

      // Upper triangular part
      const auto nAbove = rowEnd - diagPtrs[row] - 1;
      for (PetscInt k = 0; k < nAbove; ++k)
        sum -= matvals[diagPtrs[row] + 1 + k] * sampleArr[colptr[diagPtrs[row] + 1 + k]];

      // Update sample
      sampleArr[row] = (1 - omega) * sampleArr[row] + randArr[row] + invDiagArr[row] * sum;
    };

    if (sweepType == GibbsSweepType::Forward || sweepType == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linearOperator.getColoring().forEachColor([&](auto /*i*/, const auto &colorIndices) {
        for (auto idx : colorIndices)
          gibbsKernel(idx);
      });
    }

    if (sweepType == GibbsSweepType::Symmetric) {
      PetscCall(fillVecRand(randVec, randVecSize, engine));
      PetscCall(VecPointwiseMult(randVec, randVec, sqrtDiagOmega));
      PetscCall(VecAXPY(randVec, 1., rhs));
      PetscCall(VecPointwiseMult(randVec, randVec, invDiagOmega));
    }

    if (sweepType == GibbsSweepType::Backward || sweepType == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linearOperator.getColoring().forEachColorReverse([&](auto /*i*/, const auto &colorIndices) {
        for (auto idx = colorIndices.crbegin(); idx != colorIndices.crend(); ++idx)
          gibbsKernel(*idx);
      });
    }

    PetscCall(VecRestoreArrayRead(invDiagOmega, &invDiagArr));
    PetscCall(VecRestoreArray(sample, &sampleArr));
    PetscCall(VecRestoreArrayRead(randVec, &randArr));

    PetscCall(PetscHelper::endGibbsEvent());

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode gibbsRbMpi(Vec sample, Vec rhs) {
    PetscFunctionBeginUser;

    PetscCall(PetscHelper::beginGibbsEvent());

    PetscCall(fillVecRand(randVec, randVecSize, engine));
    PetscCall(VecPointwiseMult(randVec, randVec, sqrtDiagOmega));
    PetscCall(VecAXPY(randVec, 1., rhs));
    PetscCall(VecPointwiseMult(randVec, randVec, invDiagOmega));

    Mat ad, ao;
    PetscCall(MatMPIAIJGetSeqAIJ(linearOperator.getMat(), &ad, &ao, nullptr));

    const PetscInt *rowptr, *colptr;
    PetscReal *matvals;
    PetscCall(MatSeqAIJGetCSRAndMemType(ad, &rowptr, &colptr, &matvals, nullptr));

    const PetscInt *bRowptr, *bColptr;
    PetscReal *bMatvals;
    PetscCall(MatSeqAIJGetCSRAndMemType(ao, &bRowptr, &bColptr, &bMatvals, nullptr));

    PetscInt rows;
    PetscCall(MatGetSize(ad, &rows, nullptr));

    PetscReal *sampleArr;
    const PetscReal *randArr, *invDiagArr, *ghostArr;

    PetscCall(VecGetArrayRead(invDiagOmega, &invDiagArr));
    PetscCall(VecGetArrayRead(randVec, &randArr));

    MatInfo matinfo;
    PetscCall(MatGetInfo(linearOperator.getMat(), MAT_LOCAL, &matinfo));

    const auto gibbsKernel = [&](PetscInt row) {
      const auto rowStart = rowptr[row];
      const auto rowEnd = rowptr[row + 1];
      const auto rowDiag = diagPtrs[row];

      PetscReal sum = 0.;

      // Lower triangular part
      for (PetscInt k = rowStart; k < rowDiag; ++k)
        sum -= matvals[k] * sampleArr[colptr[k]];

      // Upper triangular part
      for (PetscInt k = rowDiag + 1; k < rowEnd; ++k)
        sum -= matvals[k] * sampleArr[colptr[k]];

      for (PetscInt k = 0; k < bRowptr[row + 1] - bRowptr[row]; ++k)
        sum -= bMatvals[bRowptr[row] + k] * ghostArr[k];

      // Update sample
      sampleArr[row] = (1 - omega) * sampleArr[row] + randArr[row] + invDiagArr[row] * sum;
    };

    PetscInt firstRow, lastRow;
    PetscCall(MatGetOwnershipRange(linearOperator.getMat(), &firstRow, &lastRow));

    if (sweepType == GibbsSweepType::Forward || sweepType == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linearOperator.getColoring().forEachColor([&](auto i, const auto &colorIndices) {
        PetscFunctionBeginUser;

        auto scatter = linearOperator.getColoring().getScatter(i);
        auto ghostvec = linearOperator.getColoring().getGhostVec(i);

        PetscCall(VecScatterBegin(scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));

        PetscCall(VecGetArrayRead(ghostvec, &ghostArr));
        PetscCall(VecGetArray(sample, &sampleArr));

        for (auto idx : colorIndices)
          gibbsKernel(idx);

        PetscCall(VecRestoreArray(sample, &sampleArr));
        PetscCall(VecRestoreArrayRead(ghostvec, &ghostArr));

        PetscFunctionReturn(PETSC_SUCCESS);
      });
    }

    if (sweepType == GibbsSweepType::Symmetric) {
      PetscCall(fillVecRand(randVec, randVecSize, engine));
      PetscCall(VecPointwiseMult(randVec, randVec, sqrtDiagOmega));
      PetscCall(VecAXPY(randVec, 1., rhs));
      PetscCall(VecPointwiseMult(randVec, randVec, invDiagOmega));
    }

    if (sweepType == GibbsSweepType::Backward || sweepType == GibbsSweepType::Symmetric) {
      PetscCall(PetscLogFlops(2.0 * matinfo.nz_used));

      linearOperator.getColoring().forEachColorReverse([&](auto i, const auto &colorIndices) {
        PetscFunctionBeginUser;

        auto scatter = linearOperator.getColoring().getScatter(i);
        auto ghostvec = linearOperator.getColoring().getGhostVec(i);

        PetscCall(VecScatterBegin(scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter, sample, ghostvec, INSERT_VALUES, SCATTER_FORWARD));

        PetscCall(VecGetArrayRead(ghostvec, &ghostArr));
        PetscCall(VecGetArray(sample, &sampleArr));

        for (auto idx = colorIndices.crbegin(); idx != colorIndices.crend(); ++idx)
          gibbsKernel(*idx);

        PetscCall(VecRestoreArray(sample, &sampleArr));
        PetscCall(VecRestoreArrayRead(ghostvec, &ghostArr));

        PetscFunctionReturn(PETSC_SUCCESS);
      });
    }

    PetscCall(VecRestoreArrayRead(randVec, &randArr));
    PetscCall(VecRestoreArrayRead(invDiagOmega, &invDiagArr));

    PetscCall(PetscHelper::endGibbsEvent());

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  LinearOperator &linearOperator;

  Engine &engine;
  PetscReal omega;

  Vec sqrtDiagOmega; // = sqrt((2-w)/w) * D^(1/2)
  Vec invDiagOmega;  // = w * D^-1

  Vec randVec;
  PetscInt randVecSize;

  GibbsSweepType sweepType;

  /// Only used in parallel execution
  std::vector<PetscInt> diagPtrs; // Indices of the diagonal entries
};

} // namespace parmgmc
