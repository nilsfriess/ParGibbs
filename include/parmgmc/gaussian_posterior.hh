#pragma once

#include "parmgmc/common/types.hh"
#include "parmgmc/grid/grid_operator.hh"
#include "parmgmc/samplers/sor.hh"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi_proto.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <vector>
namespace parmgmc {
struct Observation {
  Coordinate coord;
  PetscReal value;
};

/* Represents the (Gaussian) posterior distribution that arises from a linear
   Gaussian Bayesian inverse problem. Let f(x) = Ax be the observation operator.
   We consider noisy observations y = Ax + e, where e ~ N(0, S) is additive
   Gaussian noise independent of x. If we endow x with a Gaussian prior N(m, C)
   then the posterior of y|x is also Gaussian with mean

     mu = m + C A^T ( S + A C A^T )^-1 (y - A m)

   and precision matrix

     B = C^-1 + A^T S^-1 A.

   Currently, only supports Gaussian noise with diagonal covariance matrix.
   Further, the observation operator A is implicitly defined by providing a
   vector of observations, each having a coordinate corresponding to a grid
   point and a the actual measured value.
 */
template <class Engine> class GaussianPosterior {
public:
  GaussianPosterior(std::shared_ptr<GridOperator> prior_precision_operator,
                    Vec prior_mean,
                    const std::vector<Observation> &observations,
                    Vec noise_diag, Engine *engine)
      : prior_precision_operator{prior_precision_operator},
        prior_mean{prior_mean} {
    PetscFunctionBeginUser;

    PetscCallVoid(VecDuplicate(noise_diag, &noise_diag_inv));
    PetscCallVoid(VecCopy(noise_diag, noise_diag_inv));
    PetscCallVoid(VecReciprocal(noise_diag_inv));

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    PetscInt local_size, global_size;
    PetscCallVoid(VecGetLocalSize(prior_mean, &local_size));
    PetscCallVoid(VecGetSize(prior_mean, &global_size));

    std::vector<std::pair<PetscInt, PetscInt>> own_obs;
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto &obs = observations[i];

      // Find coordinates of grid point closests to the given observation
      PetscInt II, JJ;
      PetscReal XX, YY;
      PetscCallVoid(DMDAGetLogicalCoordinate(prior_precision_operator->dm,
                                             obs.coord.x,
                                             obs.coord.y,
                                             0,
                                             &II,
                                             &JJ,
                                             NULL,
                                             &XX,
                                             &YY,
                                             NULL));

      if (II == -1 || JJ == -1) // point is not on current processor
        continue;

      const auto global_idx = II * prior_precision_operator->global_x + JJ;
      own_obs.push_back({(PetscInt)i, global_idx});
    }

    // Construct lowrank factor needed in Gibbs sampler: A^T S^{-1/2}
    Mat lowrank_factor;
    PetscCallVoid(MatCreate(MPI_COMM_WORLD, &lowrank_factor));
    PetscCallVoid(MatSetSizes(lowrank_factor,
                              local_size,
                              PETSC_DECIDE,
                              global_size,
                              observations.size()));
    PetscCallVoid(MatSetType(lowrank_factor, MATAIJ));

    for (const auto &[obs_idx, obs_coord] : own_obs)
      PetscCallVoid(
          MatSetValue(lowrank_factor, obs_coord, obs_idx, 1., INSERT_VALUES));
    PetscCallVoid(MatAssemblyBegin(lowrank_factor, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(lowrank_factor, MAT_FINAL_ASSEMBLY));
    // Store A^T separately as it is needed in the computation of the exact mean
    // TODO: This can be avoided if we compute the exact mean here.
    PetscCallVoid(
        MatConvert(lowrank_factor, MATSAME, MAT_INITIAL_MATRIX, &obs_mat));

    PetscCallVoid(MatCreateVecs(lowrank_factor, &obs_values, NULL));
    for (std::size_t i = 0; i < observations.size(); ++i)
      PetscCallVoid(
          VecSetValue(obs_values, i, observations[i].value, INSERT_VALUES));
    PetscCallVoid(VecAssemblyBegin(obs_values));
    PetscCallVoid(VecAssemblyEnd(obs_values));

    Vec noise_diag_copy;
    PetscCallVoid(VecDuplicate(noise_diag, &noise_diag_copy));
    PetscCallVoid(VecCopy(noise_diag, noise_diag_copy));
    PetscCallVoid(VecSqrtAbs(noise_diag_copy));
    PetscCallVoid(VecReciprocal(noise_diag_copy));
    PetscCallVoid(MatDiagonalScale(lowrank_factor, NULL, noise_diag_copy));

    sampler = std::make_unique<SORSampler<Engine>>(
        prior_precision_operator, lowrank_factor, engine);

    // To generate samples with target mean m, we need to use C*m as the rhs in
    // the sampler (cf. [Fox, Parker, 2014]).
    Vec tmp;
    PetscCallVoid(VecDuplicate(prior_mean, &rhs));
    PetscCallVoid(VecDuplicate(prior_mean, &tmp));

    PetscCallVoid(exact_mean(tmp));
    PetscCallVoid(MatMult(prior_precision_operator->mat, tmp, rhs));

    // PetscCallVoid(MatDestroy(&lowrank_factor));
    PetscCallVoid(VecDestroy(&tmp));
    PetscCallVoid(VecDestroy(&noise_diag_copy));
    // PetscCallVoid(MatDestroy(&lowrank_factor));

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample) {
    PetscFunctionBeginUser;

    sampler->sample(sample, rhs);

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode exact_mean(Vec exact_mean) {
    PetscFunctionBeginUser;

    if (exact_mean_computed) {
      PetscCall(VecCopy(exact_mean_, exact_mean));
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(VecZeroEntries(exact_mean));

    Vec z;
    PetscCall(VecDuplicate(obs_values, &z));
    PetscCall(MatMultTransposeAdd(obs_mat, prior_mean, obs_values, z));
    PetscCall(VecScale(z, -1));

    PetscInt noise_size;
    PetscCall(VecGetSize(noise_diag_inv, &noise_size));

    Mat noise_inv;
    PetscCall(MatCreate(MPI_COMM_WORLD, &noise_inv));
    PetscCall(MatSetSizes(
        noise_inv, PETSC_DECIDE, PETSC_DECIDE, noise_size, noise_size));
    PetscCall(MatSetUp(noise_inv));
    PetscCall(MatDiagonalSet(noise_inv, noise_diag_inv, INSERT_VALUES));

    Mat obs_noise_inv; // = A^T S^-1
    PetscCall(MatMatMult(
        obs_mat, noise_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &obs_noise_inv));

    PetscCall(MatMult(obs_noise_inv, z, exact_mean));

    Mat Q_AtSA;
    PetscCall(MatMatTransposeMult(
        obs_noise_inv, obs_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Q_AtSA));

    PetscCall(MatAXPY(
        Q_AtSA, 1., prior_precision_operator->mat, UNKNOWN_NONZERO_PATTERN));

    {
      KSP ksp;
      PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
      PetscCall(KSPSetOperators(ksp, Q_AtSA, Q_AtSA));
      PetscCall(KSPSolve(ksp, exact_mean, exact_mean));
      PetscCall(KSPDestroy(&ksp));
    }

    Vec tmp1, tmp2;
    PetscCall(VecDuplicate(z, &tmp1));
    PetscCall(VecDuplicate(z, &tmp2));
    PetscCall(MatMultTranspose(obs_noise_inv, exact_mean, tmp1));

    PetscCall(MatMult(noise_inv, z, tmp2));

    PetscCall(VecAXPY(tmp1, -1, tmp2));

    PetscCall(MatMult(obs_mat, tmp1, exact_mean));

    {
      KSP ksp;
      PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
      PetscCall(KSPSetOperators(
          ksp, prior_precision_operator->mat, prior_precision_operator->mat));
      PetscCall(KSPSolve(ksp, exact_mean, exact_mean));
      PetscCall(KSPDestroy(&ksp));
    }

    PetscCall(VecAXPY(exact_mean, 1., prior_mean));

    // Cache exact_mean
    PetscCall(VecDuplicate(exact_mean, &exact_mean_));
    PetscCall(VecCopy(exact_mean, exact_mean_));
    exact_mean_computed = true;

    PetscCall(MatDestroy(&obs_noise_inv));
    PetscCall(MatDestroy(&noise_inv));
    PetscCall(MatDestroy(&Q_AtSA));
    PetscCall(VecDestroy(&tmp1));
    PetscCall(VecDestroy(&tmp2));
    PetscCall(VecDestroy(&z));

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~GaussianPosterior() {
    VecDestroy(&noise_diag_inv);
    VecDestroy(&rhs);
    VecDestroy(&obs_values);
    VecDestroy(&exact_mean_);
    MatDestroy(&obs_mat);
  }

  GaussianPosterior(GaussianPosterior &&other)
      : prior_precision_operator(std::move(other.prior_precision_operator)),
        sampler(std::move(other.sampler)), rhs(other.rhs),
        noise_diag_inv(other.noise_diag_inv), prior_mean(other.prior_mean),
        obs_mat(other.obs_mat), obs_values(other.obs_values),
        exact_mean_(other.exact_mean_),
        exact_mean_computed(other.exact_mean_computed) {}

private:
  std::shared_ptr<GridOperator> prior_precision_operator;
  std::unique_ptr<SORSampler<Engine>> sampler;

  Vec rhs;

  Vec noise_diag_inv;

  Vec prior_mean;

  Mat obs_mat;
  Vec obs_values;

  Vec exact_mean_;
  bool exact_mean_computed = false;
};
} // namespace parmgmc
