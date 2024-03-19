#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/common/timer.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"
#include <array>

#include <memory>
#include <mpi.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscerror.h>
#include <petscsystypes.h>
#include <random>

using namespace parmgmc;

class ShiftedLaplaceFD {
public:
  ShiftedLaplaceFD(PetscInt verticesPerDim, PetscReal kappainv = 1.,
                   bool colorMatrixWithDM = true) {
    PetscFunctionBeginUser;

    PetscCallVoid(DMDACreate2d(MPI_COMM_WORLD,
                               DM_BOUNDARY_NONE,
                               DM_BOUNDARY_NONE,
                               DMDA_STENCIL_STAR,
                               verticesPerDim,
                               verticesPerDim,
                               PETSC_DECIDE,
                               PETSC_DECIDE,
                               1,
                               1,
                               nullptr,
                               nullptr,
                               &da));
    PetscCallVoid(DMSetUp(da));
    PetscCallVoid(DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0));

    Mat mat;
    PetscCallVoid(DMCreateMatrix(da, &mat));

    // TODO: Maybe not needed?
    PetscCallVoid(MatSetOption(mat, MAT_USE_INODES, PETSC_FALSE));

    // Assemble matrix
    MatStencil row;
    std::array<MatStencil, 5> cols;
    std::array<PetscReal, 5> vals;

    double h2inv = 1. / ((verticesPerDim - 1) * (verticesPerDim - 1));

    dirichletRows.reserve(4 * verticesPerDim);

    DMDALocalInfo info;
    PetscCallVoid(DMDAGetLocalInfo(da, &info));

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

          PetscCallVoid(MatSetValuesStencil(
              mat, 1, &row, k, cols.data(), vals.data(), INSERT_VALUES));
        }
      }
    }

    PetscCallVoid(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

    // Dirichlet rows are in natural ordering, convert to global using the DM's
    // ApplicationOrdering
    AO ao;
    PetscCallVoid(DMDAGetAO(da, &ao));
    PetscCallVoid(
        AOApplicationToPetsc(ao, dirichletRows.size(), dirichletRows.data()));

    PetscCallVoid(MatZeroRowsColumns(
        mat, dirichletRows.size(), dirichletRows.data(), 1., nullptr, nullptr));

    PetscCallVoid(MatSetOption(mat, MAT_SPD, PETSC_TRUE));

    op = std::make_shared<LinearOperator>(mat, true);
    if (colorMatrixWithDM)
      op->color_matrix(da);
    else
      op->color_matrix();

    PetscFunctionReturnVoid();
  }

  const std::shared_ptr<LinearOperator> &getOperator() const { return op; }

  const std::vector<PetscInt> &getDirichletRows() const {
    return dirichletRows;
  }

  ~ShiftedLaplaceFD() {
    PetscFunctionBeginUser;

    PetscCallVoid(DMDestroy(&da));

    PetscFunctionReturnVoid();
  }

private:
  std::shared_ptr<LinearOperator> op;
  DM da;

  std::vector<PetscInt> dirichletRows;
};

struct TimingResult {
  double setupTime = 0;
  double sampleTime = 0;
};

template <typename Engine>
PetscErrorCode testGibbsSampler(const ShiftedLaplaceFD &problem,
                                PetscInt n_samples, Engine &engine,
                                PetscScalar omega, GibbsSweepType sweepType,
                                bool fixRhs, TimingResult &timingResult) {
  PetscFunctionBeginUser;

  Vec sample, rhs;
  PetscCall(MatCreateVecs(problem.getOperator()->get_mat(), &sample, nullptr));
  PetscCall(VecDuplicate(sample, &rhs));

  PetscCall(MatZeroRowsColumns(problem.getOperator()->get_mat(),
                               problem.getDirichletRows().size(),
                               problem.getDirichletRows().data(),
                               1.,
                               sample,
                               rhs));

  PetscCall(fill_vec_rand(rhs, engine));

  Timer timer;

  // Measure setup time
  timer.reset();
  MulticolorGibbsSampler sampler(
      problem.getOperator(), &engine, omega, sweepType);

  if (fixRhs)
    sampler.setFixedRhs(rhs);

  timingResult.setupTime = timer.elapsed();
  // Setup done

  // Measure sample time
  timer.reset();
  for (PetscInt n = 0; n < n_samples; ++n)
    PetscCall(sampler.sample(sample, rhs));

  timingResult.sampleTime = timer.elapsed();
  // Sampling done

  // Cleanup
  PetscCall(VecDestroy(&sample));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResult(const std::string &name, TimingResult timing) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "\n+++-------------------------------------------------"
                        "-----------+++\n\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Name: %s\n", name.c_str()));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Timing:\n"));
  PetscCall(PetscPrintf(
      MPI_COMM_WORLD, "   Setup time:    %.4fs\n", timing.setupTime));
  PetscCall(PetscPrintf(
      MPI_COMM_WORLD, "   Sampling time: %.4fs\n", timing.sampleTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "   -----------------------\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "   Total:         %.4fs\n",
                        timing.setupTime + timing.sampleTime));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "\n+++-------------------------------------------------"
                        "-----------+++\n"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscFunctionBeginUser;

  PetscInt size = 9;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-size", &size, nullptr));
  PetscInt n_samples = 1000;
  PetscCall(
      PetscOptionsGetInt(nullptr, nullptr, "-samples", &n_samples, nullptr));

  PetscMPIInt mpisize;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &mpisize));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "##################################################"
                        "################\n"));
  PetscCall(PetscPrintf(
      MPI_COMM_WORLD,
      "####            Running strong scaling test suite           ######\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "##################################################"
                        "################\n"));
  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "Configuration: \n\tMPI rank(s):  %d\n\tProblem size: "
                        "%dx%d = %d\n\tSamples:      %d\n",
                        mpisize,
                        size,
                        size,
                        (size * size),
                        n_samples));

  ShiftedLaplaceFD problem(size);

  std::mt19937 engine;

  {
    TimingResult timing;
    PetscCall(testGibbsSampler(
        problem, n_samples, engine, 1., GibbsSweepType::Forward, true, timing));

    PetscCall(printResult("Gibbs sampler, forward sweep, fixed rhs", timing));
  }

  // {
  //   TimingResult timing;
  //   PetscCall(testGibbsSampler(problem,
  //                              n_samples,
  //                              engine,
  //                              1.,
  //                              GibbsSweepType::Forward,
  //                              false,
  //                              timing));

  //   PetscCall(
  //       printResult("Gibbs sampler, forward sweep, nonfixed rhs", timing));
  // }

  // {
  //   TimingResult timing;
  //   PetscCall(testGibbsSampler(problem,
  //                              n_samples,
  //                              engine,
  //                              1.,
  //                              GibbsSweepType::Symmetric,
  //                              true,
  //                              timing));

  //   PetscCall(
  //       printResult("Gibbs sampler, symmetric sweep, nonfixed rhs", timing));
  // }
}
