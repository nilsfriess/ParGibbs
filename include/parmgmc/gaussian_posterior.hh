
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
   Gaussian noise independent of x. If we endow x with a Gaussian prior N(m,
   C^-1) then the posterior of y|x is also Gaussian with mean

     mu = m + C^-1 A^T ( S + A C^-1 A^T )^-1 (y - A m)

   and precision matrix

     B = C + A^T S^-1 A.

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

    PetscInt local_size, global_size;
    PetscCallVoid(VecGetLocalSize(prior_mean, &local_size));
    PetscCallVoid(VecGetSize(prior_mean, &global_size));

    { // Store S^{-1} as vector
      PetscCallVoid(VecDuplicate(noise_diag, &noise_diag_inv));
      PetscCallVoid(VecCopy(noise_diag, noise_diag_inv));
      PetscCallVoid(VecReciprocal(noise_diag_inv));
    }

    { // Setup observation matrix A
      std::vector<std::pair<PetscInt, PetscInt>> own_obs;
      PetscCallVoid(get_own_observations(observations, own_obs));

      PetscCallVoid(MatCreate(MPI_COMM_WORLD, &obs_mat));
      PetscCallVoid(MatSetSizes(
          obs_mat, PETSC_DECIDE, local_size, observations.size(), global_size));
      PetscCallVoid(MatSetType(obs_mat, MATAIJ));

      // Set A_ij = 1 if observation j is closest to grid index i
      for (const auto &[obs_idx, obs_coord] : own_obs)
        PetscCallVoid(
            MatSetValue(obs_mat, obs_idx, obs_coord, 1., INSERT_VALUES));

      PetscCallVoid(MatAssemblyBegin(obs_mat, MAT_FINAL_ASSEMBLY));
      PetscCallVoid(MatAssemblyEnd(obs_mat, MAT_FINAL_ASSEMBLY));
    }

    { // Store observation values in vec
      PetscCallVoid(MatCreateVecs(obs_mat, NULL, &obs_values));
      for (std::size_t i = 0; i < observations.size(); ++i)
        PetscCallVoid(
            VecSetValue(obs_values, i, observations[i].value, INSERT_VALUES));
      PetscCallVoid(VecAssemblyBegin(obs_values));
      PetscCallVoid(VecAssemblyEnd(obs_values));
    }

    { // Create lowrank update A^T S^{-1} A and attach to grid operator
      Vec noise_inv_sqrt;
      PetscCallVoid(VecDuplicate(noise_diag_inv, &noise_inv_sqrt));
      PetscCallVoid(VecCopy(noise_diag_inv, noise_inv_sqrt));
      PetscCallVoid(VecSqrtAbs(noise_inv_sqrt));

      prior_precision_operator->set_lowrank_factor(
          obs_mat, noise_diag_inv, noise_inv_sqrt);
    }

    { // Set potential vector
      Vec posterior_mean;
      PetscCallVoid(VecDuplicate(prior_mean, &posterior_mean));
      PetscCallVoid(exact_mean(posterior_mean));

      PetscCallVoid(VecDuplicate(posterior_mean, &rhs));

      PetscCallVoid(prior_precision_operator->apply(posterior_mean, rhs));
      
      PetscCallVoid(VecDestroy(&posterior_mean));
    }

    sampler =
        std::make_unique<SORSampler<Engine>>(prior_precision_operator, engine);

    PetscFunctionReturnVoid();
  }

  PetscErrorCode sample(Vec sample) {
    PetscFunctionBeginUser;

    sampler->sample(sample, rhs);

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* The exact mean is given by
         mu = m + C^-1 A^T ( S + A C^-1 A^T )^-1 (y - A m).
     To avoid the computation of the (inner) inverse of the precision matrix C,
     we use the Woodbury matrix identity and rewrite mu as
         mu = m + C^-1 A^T (S^-1 - S^-1 A ( C + A^T S^-1 A)^-1 A^T S) (y - A m).
   */
  PetscErrorCode exact_mean(Vec exact_mean) {
    PetscFunctionBeginUser;

    if (exact_mean_computed) {
      PetscCall(VecCopy(exact_mean_, exact_mean));
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(VecZeroEntries(exact_mean));

    Vec z1;
    { // z1 = y - A m (computed as z1 = -(-y + Am))
      PetscCall(VecDuplicate(obs_values, &z1));

      PetscCall(VecScale(obs_values, -1)); // y <-- -y
      PetscCall(MatMultAdd(obs_mat, prior_mean, obs_values, z1));
      PetscCall(VecScale(obs_values, -1)); // y <-- -y

      PetscCall(VecScale(z1, -1));
    }

    Mat AtSinv;
    { // AtS =  A^T S^-1
      PetscCall(MatTranspose(obs_mat, MAT_INITIAL_MATRIX, &AtSinv));
      PetscCall(MatDiagonalScale(AtSinv, NULL, noise_diag_inv));
    }

    Vec tmp;

    { // tmp = A^T S^-1 (y - Am) = AtSinv * z1
      PetscCall(VecDuplicate(exact_mean, &tmp));
      PetscCall(MatMult(AtSinv, z1, tmp));
    }

    { // tmp = (C + A^T S^-1 A)^-1 * A^T S^-1 (y - Am)
      //     = (C + A^T S^-1 A)^-1 * tmp
      Mat Sinv;

      PetscInt global_size;
      PetscCall(VecGetSize(noise_diag_inv, &global_size));

      PetscCall(MatCreate(MPI_COMM_WORLD, &Sinv));
      // PetscCall(MatSetType(Sinv, MATDIAGONAL));
      PetscCall(MatSetType(Sinv, MATAIJ));
      PetscCall(MatSetSizes(
          Sinv, PETSC_DECIDE, PETSC_DECIDE, global_size, global_size));
      PetscCall(MatSetUp(Sinv));
      PetscCall(MatDiagonalSet(Sinv, noise_diag_inv, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(Sinv, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Sinv, MAT_FINAL_ASSEMBLY));

      Mat M;
      PetscCall(MatPtAP(Sinv, obs_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M));
      PetscCall(MatAXPY(
          M, 1., prior_precision_operator->mat, UNKNOWN_NONZERO_PATTERN));

      KSP ksp;
      PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPSetOperators(ksp, M, M));
      PetscCall(KSPSolve(ksp, tmp, tmp));

      PetscCall(KSPDestroy(&ksp));
      PetscCall(MatDestroy(&M));
    }

    Vec z2;
    { // z2 = S^-1 A * tmp
      PetscCall(VecDuplicate(z1, &z2));
      PetscCall(MatMult(obs_mat, tmp, z2));
      PetscCall(VecPointwiseMult(z2, z2, noise_diag_inv));
    }

    { // z1 = S^-1 z1
      PetscCall(VecPointwiseMult(z1, z1, noise_diag_inv));
    }

    { // z1 = z1 - z2
      PetscCall(VecAXPY(z1, -1., z2));
    }

    { // exact_mean = C^-1 A^T z1
      PetscCall(MatMultTranspose(obs_mat, z1, exact_mean));

      KSP ksp;
      PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPSetOperators(
          ksp, prior_precision_operator->mat, prior_precision_operator->mat));
      PetscCall(KSPSolve(ksp, exact_mean, exact_mean));

      PetscCall(KSPDestroy(&ksp));
    }

    { // exact_mean += prior_mean
      PetscCall(VecAXPY(exact_mean, 1., prior_mean));
    }

    { // Cleanup
      PetscCall(VecDestroy(&z2));
      PetscCall(VecDestroy(&tmp));
      PetscCall(MatDestroy(&AtSinv));
      PetscCall(VecDestroy(&z1));
    }

    { // Cache computed posterior mean
      PetscCall(VecDuplicate(exact_mean, &exact_mean_));
      PetscCall(VecCopy(exact_mean, exact_mean_));
      exact_mean_computed = true;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ~GaussianPosterior() {
    VecDestroy(&noise_diag_inv);
    VecDestroy(&rhs);
    VecDestroy(&obs_values);
    // VecDestroy(&exact_mean_);
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
  PetscErrorCode
  get_own_observations(const std::vector<Observation> &observations,
                       std::vector<std::pair<PetscInt, PetscInt>> &own_obs) {
    PetscFunctionBeginUser;

    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto &obs = observations[i];

      PetscInt II, JJ;
      PetscReal XX, YY;
      PetscCall(DMDAGetLogicalCoordinate(prior_precision_operator->dm,
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

    PetscFunctionReturn(PETSC_SUCCESS);
  }

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
