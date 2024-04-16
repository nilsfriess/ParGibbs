#pragma once

#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"

#include <array>
#include <memory>
#include <petscdm.h>
#include <stdexcept>

#include <mpi.h>

#include <petscdmda.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsystypes.h>

struct Dim {
  explicit Dim(PetscInt d) : d{d} {}
  operator PetscInt() { return d; }

  PetscInt d;
};

class Problem {
public:
  Problem(Dim dim, PetscInt initialVerticesPerDim, PetscInt levels, bool sizeIsFine = false) {
    PetscFunctionBeginUser;

    if (!(dim == 2 || dim == 3))
      throw std::runtime_error("Only dim == 2 or dim == 3 supported");

    // Create coarse DM
    DM da;
    if (dim == 2)
      PetscCallVoid(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                                 DMDA_STENCIL_STAR, initialVerticesPerDim, initialVerticesPerDim,
                                 PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &da));
    else
      PetscCallVoid(DMDACreate3d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                                 DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, initialVerticesPerDim,
                                 initialVerticesPerDim, initialVerticesPerDim, PETSC_DECIDE,
                                 PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &da));

    PetscCallVoid(DMSetUp(da));
    PetscCallVoid(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));

    auto dmtype = sizeIsFine ? parmgmc::DMInitialType::Finest : parmgmc::DMInitialType::Coarsest;
    hierarchy = parmgmc::DMHierarchy{da, levels, true, dmtype};

    PetscFunctionReturnVoid();
  }

  [[nodiscard]] std::shared_ptr<parmgmc::LinearOperator> getFineOperator() const {
    if (ops.size() == 1)
      return ops[0];
    else
      return ops[hierarchy.numLevels() - 1];
  }
  [[nodiscard]] const parmgmc::DMHierarchy &getHierarchy() const { return hierarchy; }

  [[nodiscard]] DM getCoarseDM() const { return hierarchy.getCoarse(); }
  [[nodiscard]] DM getFineDM() const { return hierarchy.getFine(); }

  virtual std::string getName() const = 0;

  virtual ~Problem() = default;

protected:
  std::vector<std::shared_ptr<parmgmc::LinearOperator>> ops;
  parmgmc::DMHierarchy hierarchy;
};

class DiagonalPrecisionMatrix : public Problem {
public:
  DiagonalPrecisionMatrix(Dim dim, PetscInt initialVerticesPerDim, PetscInt levels)
      : Problem{dim, initialVerticesPerDim, levels} {
    PetscFunctionBeginUser;

    // Create matrix corresponding to operator on fine DM
    Mat mat;
    PetscCallVoid(DMCreateMatrix(hierarchy.getFine(), &mat));

    // Assemble matrix
    Vec diag;
    PetscCallVoid(MatCreateVecs(mat, &diag, nullptr));
    // auto engine = std::mt19937{std::random_device{}()};
    // PetscCallVoid(parmgmc::fillVecRand(diag, engine));
    PetscCallVoid(VecSet(diag, 1.));
    PetscCallVoid(VecAbs(diag));
    PetscCallVoid(MatDiagonalSet(mat, diag, INSERT_VALUES));

    PetscCallVoid(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    ops.emplace_back(std::make_shared<parmgmc::LinearOperator>(mat, true));
    ops[0]->colorMatrix(hierarchy.getFine());

    PetscFunctionReturnVoid();
  }

  std::string getName() const override { return "Diagonal Precision matrix"; };
};

class ShiftedLaplaceFD : public Problem {
public:
  ShiftedLaplaceFD(Dim dim, PetscInt initialVerticesPerDim, PetscInt levels,
                   PetscReal kappainv = 1., bool colorMatrixWithDM = true, bool sizeIsFine = false)
      : Problem{dim, initialVerticesPerDim, levels, sizeIsFine} {
    PetscFunctionBeginUser;

    // Create matrix corresponding to operator on fine DM
    Mat mat;
    PetscCallVoid(DMCreateMatrix(hierarchy.getFine(), &mat));

    // Assemble matrix
    MatStencil row;
    std::array<MatStencil, 7> cols;
    std::array<PetscReal, 7> vals;

    DMDALocalInfo info;
    PetscCallVoid(DMDAGetLocalInfo(hierarchy.getFine(), &info));

    const auto kappa2 = 1. / (kappainv * kappainv);

    double h2inv = 1. / ((info.mx - 1) * (info.mx - 1));

    if (dim == 2) {
      for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
          row.j = j;
          row.i = i;

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

          PetscCallVoid(
              MatSetValuesStencil(mat, 1, &row, k, cols.data(), vals.data(), INSERT_VALUES));
        }
      }
    } else {
      for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
          for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            row.k = k;
            row.j = j;
            row.i = i;

            std::size_t n = 0;

            if (k != 0) {
              cols[n].k = k - 1;
              cols[n].j = j;
              cols[n].i = i;
              vals[n] = -h2inv;
              ++n;
            }

            if (k != info.mz - 1) {
              cols[n].k = k + 1;
              cols[n].j = j;
              cols[n].i = i;
              vals[n] = -h2inv;
              ++n;
            }

            if (j != 0) {
              cols[n].k = k;
              cols[n].j = j - 1;
              cols[n].i = i;
              vals[n] = -h2inv;
              ++n;
            }

            if (i != 0) {
              cols[n].k = k;
              cols[n].j = j;
              cols[n].i = i - 1;
              vals[n] = -h2inv;
              ++n;
            }

            cols[n].k = k;
            cols[n].j = j;
            cols[n].i = i;
            vals[n] = 6 * h2inv + kappa2;
            ++n;

            if (j != info.my - 1) {
              cols[n].k = k;
              cols[n].j = j + 1;
              cols[n].i = i;
              vals[n] = -h2inv;
              ++n;
            }

            if (i != info.mx - 1) {
              cols[n].k = k;
              cols[n].j = j;
              cols[n].i = i + 1;
              vals[n] = -h2inv;
              ++n;
            }

            PetscCallVoid(
                MatSetValuesStencil(mat, 1, &row, n, cols.data(), vals.data(), INSERT_VALUES));
          }
        }
      }
    }

    PetscCallVoid(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    ops.emplace_back(std::make_shared<parmgmc::LinearOperator>(mat, true));
    if (colorMatrixWithDM)
      ops[0]->colorMatrix(hierarchy.getFine());
    else
      ops[0]->colorMatrix();

    PetscFunctionReturnVoid();
  }

  std::string getName() const override { return "Shifted Laplace"; };
};

class SimpleGMRF : public Problem {
public:
  SimpleGMRF(Dim dim, PetscInt initialVerticesPerDim, PetscInt levels,
             bool colorMatrixWithDM = true)
      : Problem{dim, initialVerticesPerDim, levels} {
    PetscFunctionBeginUser;

    if (dim != 2)
      throw std::runtime_error("Only dim == 2");

    // Create matrix corresponding to operator on fine DM
    Mat mat;
    PetscCallVoid(DMCreateMatrix(hierarchy.getFine(), &mat));

    // Assemble matrix
    MatStencil row;
    std::array<MatStencil, 5> cols;
    std::array<PetscReal, 5> vals;

    DMDALocalInfo info;
    PetscCallVoid(DMDAGetLocalInfo(hierarchy.getFine(), &info));

    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
      for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
        row.j = j;
        row.i = i;

        std::size_t k = 0;

        if (j != 0) {
          cols[k].j = j - 1;
          cols[k].i = i;
          vals[k] = -1;
          ++k;
        }

        if (i != 0) {
          cols[k].j = j;
          cols[k].i = i - 1;
          vals[k] = -1;
          ++k;
        }

        if (j != info.my - 1) {
          cols[k].j = j + 1;
          cols[k].i = i;
          vals[k] = -1;
          ++k;
        }

        if (i != info.mx - 1) {
          cols[k].j = j;
          cols[k].i = i + 1;
          vals[k] = -1;
          ++k;
        }

        cols[k].j = j;
        cols[k].i = i;
        vals[k] = k + 1e-2;
        ++k;

        PetscCallVoid(
            MatSetValuesStencil(mat, 1, &row, k, cols.data(), vals.data(), INSERT_VALUES));
      }
    }

    PetscCallVoid(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    ops.emplace_back(std::make_shared<parmgmc::LinearOperator>(mat, true));
    if (colorMatrixWithDM)
      ops[0]->colorMatrix(hierarchy.getFine());
    else
      ops[0]->colorMatrix();

    PetscFunctionReturnVoid();
  }

  std::string getName() const override { return "Simple GMRF"; };
};
