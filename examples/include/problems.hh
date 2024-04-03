#pragma once

#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"

#include <array>
#include <memory>

#include <mpi.h>

#include <petscdmda.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscsystypes.h>
#include <stdexcept>

// inline PetscInt nextPower2(PetscInt num) {
//   PetscInt power = 1;
//   while (power < num)
//     power *= 2;
//   return power;
// }

struct Dim {
  explicit Dim(PetscInt d) : d{d} {}
  operator PetscInt() { return d; }

  PetscInt d;
};

class ShiftedLaplaceFD {
public:
  ShiftedLaplaceFD(Dim dim, PetscInt globalCoarseVerticesPerDim, PetscInt refineLevels,
                   PetscReal kappainv = 1., bool colorMatrixWithDM = true,
                   bool includeDirichletRows = false) {
    PetscFunctionBeginUser;

    if (!(dim == 2 || dim == 3))
      throw std::runtime_error("Only dim == 2 or dim == 3 supported");

    // PetscInt globalVerticesPerDim;
    // if (strong) {
    //   globalVerticesPerDim = coarseVerticesPerDim;
    // } else {
    //   PetscMPIInt mpisize;
    //   MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    //   PetscInt localSize = coarseVerticesPerDim * coarseVerticesPerDim;
    //   PetscInt targetGlobalSize = mpisize * localSize;

    //   globalVerticesPerDim = nextPower2((unsigned int)std::sqrt(targetGlobalSize)) + 1;
    // }

    // Create coarse DM
    DM da;
    if (dim == 2)
      PetscCallVoid(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                                 DMDA_STENCIL_STAR, globalCoarseVerticesPerDim,
                                 globalCoarseVerticesPerDim, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                                 nullptr, nullptr, &da));
    else
      PetscCallVoid(DMDACreate3d(
          MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
          globalCoarseVerticesPerDim, globalCoarseVerticesPerDim, globalCoarseVerticesPerDim,
          PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &da));

    PetscCallVoid(DMSetUp(da));
    PetscCallVoid(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 1));

    // Create hierarchy
    hierarchy = parmgmc::DMHierarchy{da, refineLevels, true};

    // Create matrix corresponding to operator on fine DM
    Mat mat;
    PetscCallVoid(DMCreateMatrix(hierarchy.getFine(), &mat));

    // Assemble matrix
    MatStencil row;
    std::array<MatStencil, 7> cols;
    std::array<PetscReal, 7> vals;

    DMDALocalInfo info;
    PetscCallVoid(DMDAGetLocalInfo(hierarchy.getFine(), &info));

    dirichletRows.reserve(6 * info.mx);
    const auto kappa2 = 1. / (kappainv * kappainv);

    double h2inv = 1. / ((info.mx - 1) * (info.mx - 1));

    if (dim == 2) {
      for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
          row.j = j;
          row.i = i;

          if ((i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1)) {
            if (includeDirichletRows) {
              dirichletRows.push_back(j * info.my + i);
              continue;
            }
          }
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

            if ((i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1)) {
              if (includeDirichletRows) {
                dirichletRows.push_back(k * (info.my * info.mz) + j * info.my + i);
                continue;
              }
            }
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

    // Dirichlet rows are in natural ordering, convert to global using the DM's
    // ApplicationOrdering
    if (includeDirichletRows) {
      AO ao;
      PetscCallVoid(DMDAGetAO(hierarchy.getFine(), &ao));
      PetscCallVoid(AOApplicationToPetsc(ao, dirichletRows.size(), dirichletRows.data()));

      PetscCallVoid(MatZeroRowsColumns(mat, dirichletRows.size(), dirichletRows.data(), 1., nullptr,
                                       nullptr));
    }

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    op = parmgmc::LinearOperator{mat, true};
    if (colorMatrixWithDM)
      op.colorMatrix(da);
    else
      op.colorMatrix();

    PetscFunctionReturnVoid();
  }

  [[nodiscard]] parmgmc::LinearOperator &getOperator() { return op; }
  [[nodiscard]] const parmgmc::DMHierarchy &getHierarchy() const { return hierarchy; }

  [[nodiscard]] const std::vector<PetscInt> &getDirichletRows() const { return dirichletRows; }

  [[nodiscard]] DM getCoarseDM() const { return hierarchy.getCoarse(); }
  [[nodiscard]] DM getFineDM() const { return hierarchy.getFine(); }

private:
  parmgmc::LinearOperator op;
  parmgmc::DMHierarchy hierarchy;

  std::vector<PetscInt> dirichletRows;
};
