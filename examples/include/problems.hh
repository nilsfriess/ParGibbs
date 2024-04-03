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

// inline PetscInt nextPower2(PetscInt num) {
//   PetscInt power = 1;
//   while (power < num)
//     power *= 2;
//   return power;
// }

class ShiftedLaplaceFD {
public:
  ShiftedLaplaceFD(PetscInt globalCoarseVerticesPerDim, PetscInt refineLevels,
                   PetscReal kappainv = 1., bool colorMatrixWithDM = true) {
    PetscFunctionBeginUser;

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
    PetscCallVoid(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                               DMDA_STENCIL_STAR, globalCoarseVerticesPerDim,
                               globalCoarseVerticesPerDim, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                               nullptr, nullptr, &da));
    PetscCallVoid(DMSetUp(da));
    PetscCallVoid(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0));

    // Create hierarchy
    hierarchy = std::make_shared<parmgmc::DMHierarchy>(da, refineLevels, true);

    // Create matrix corresponding to operator on fine DM
    Mat mat;
    PetscCallVoid(DMCreateMatrix(hierarchy->getFine(), &mat));

    // Assemble matrix
    MatStencil row;
    std::array<MatStencil, 5> cols;
    std::array<PetscReal, 5> vals;

    DMDALocalInfo info;
    PetscCallVoid(DMDAGetLocalInfo(hierarchy->getFine(), &info));

    dirichletRows.reserve(4 * info.mx);
    double h2inv = 1. / ((info.mx - 1) * (info.mx - 1));
    const auto kappa2 = 1. / (kappainv * kappainv);

    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
      for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
        row.j = j;
        row.i = i;

        if ((i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1)) {
          dirichletRows.push_back(j * info.my + i);
        } else {
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
    }

    PetscCallVoid(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    // Dirichlet rows are in natural ordering, convert to global using the DM's
    // ApplicationOrdering
    AO ao;
    PetscCallVoid(DMDAGetAO(hierarchy->getFine(), &ao));
    PetscCallVoid(AOApplicationToPetsc(ao, dirichletRows.size(), dirichletRows.data()));

    PetscCallVoid(
        MatZeroRowsColumns(mat, dirichletRows.size(), dirichletRows.data(), 1., nullptr, nullptr));

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    op = std::make_shared<parmgmc::LinearOperator>(mat, true);
    if (colorMatrixWithDM)
      op->colorMatrix(da);
    else
      op->colorMatrix();

    PetscFunctionReturnVoid();
  }

  [[nodiscard]] const std::shared_ptr<parmgmc::LinearOperator> &getOperator() const { return op; }
  [[nodiscard]] const std::shared_ptr<parmgmc::DMHierarchy> &getHierarchy() const {
    return hierarchy;
  }

  [[nodiscard]] const std::vector<PetscInt> &getDirichletRows() const { return dirichletRows; }

  [[nodiscard]] DM getCoarseDM() const { return hierarchy->getCoarse(); }
  [[nodiscard]] DM getFineDM() const { return hierarchy->getFine(); }

private:
  std::shared_ptr<parmgmc::LinearOperator> op;
  std::shared_ptr<parmgmc::DMHierarchy> hierarchy;

  std::vector<PetscInt> dirichletRows;
};
