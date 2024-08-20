#pragma once

#include "params.hh"
#include "parmgmc/ms.h"
#include "parmgmc/obs.h"

#include <petscdm.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

#if __has_include(<mfem.hpp>)
  #define PARMGMC_HAVE_MFEM
  #include <mfem.hpp>
#endif

struct MeasCtx {
  PetscScalar *centre, radius, vol;
};

inline PetscErrorCode f(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)time;
  (void)Nc;
  MeasCtx    *octx = (MeasCtx *)ctx;
  PetscScalar diff = 0;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < dim; ++i) diff += PetscSqr(x[i] - octx->centre[i]);
  if (diff < PetscSqr(octx->radius)) *u = 1 / octx->vol;
  else *u = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode VolumeOfSphere(DM dm, PetscScalar r, PetscScalar *v)
{
  PetscInt cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == 2 || cdim == 3, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only dim=2 and dim=3 supported");
  if (cdim == 2) *v = PETSC_PI * r * r;
  else *v = 4 * PETSC_PI / 3. * r * r * r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode CreateMatrixPetsc(Parameters params, Mat *A, Vec *meas_vec, DM *dm)
{
  MS ms;

  PetscFunctionBeginUser;
  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(MSSetFromOptions(ms));
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, A));
  PetscCall(MSGetDM(ms, dm));

  if (params->with_lr) {
    const PetscInt nobs = 3;
    PetscScalar    obs[3 * nobs], radii[nobs], obsvals[nobs], obsval = 1;
    Mat            A2, B;
    Vec            S;

    obs[0] = 0.25;
    obs[1] = 0.25;
    obs[2] = 0.75;
    obs[3] = 0.75;
    obs[4] = 0.25;
    obs[5] = 0.75;

    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-obsval", &obsval, nullptr));
    obsvals[0] = obsval;
    radii[0]   = 0.1;
    obsvals[1] = obsval;
    radii[1]   = 0.1;
    obsvals[2] = obsval;
    radii[2]   = 0.1;

    PetscCall(MakeObservationMats(*dm, nobs, 1e-6, obs, radii, obsvals, &B, &S, nullptr));
    PetscCall(MatCreateLRC(*A, B, S, nullptr, &A2));
    PetscCall(MatDestroy(&B));
    PetscCall(VecDestroy(&S));

    *A = A2;
  }
  if (!params->with_lr) PetscCall(PetscObjectReference((PetscObject)(*A)));
  PetscCall(PetscObjectReference((PetscObject)(*dm))); // Make sure MSDestroy doesn't destroy the DM because we're returning it
  PetscCall(MSDestroy(&ms));

  // Create measurement vector
  Mat      M;
  Vec      u;
  MeasCtx  ctx;
  PetscInt dim;
  void    *mctx = &ctx;

  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {f};

  PetscCall(DMGetCoordinateDim(*dm, &dim));
  PetscCall(PetscMalloc1(dim, &ctx.centre));
  for (PetscInt i = 0; i < dim; ++i) ctx.centre[i] = 0.5;
  PetscCall(VolumeOfSphere(*dm, 0.5, &ctx.vol));
  ctx.radius = 0.5;
  PetscCall(DMCreateGlobalVector(*dm, meas_vec));
  PetscCall(DMCreateMassMatrix(*dm, *dm, &M));
  PetscCall(DMGetGlobalVector(*dm, &u));
  PetscCall(DMProjectFunction(*dm, 0, funcs, &mctx, INSERT_VALUES, u));
  PetscCall(MatMult(M, u, *meas_vec));
  PetscCall(DMRestoreGlobalVector(*dm, &u));
  PetscCall(MatDestroy(&M));
  PetscCall(PetscFree(ctx.centre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifdef PARMGMC_HAVE_MFEM
class Problem {
public:
  Problem()
  {
    char      mesh_file[256];
    PetscBool flag;
    PetscOptionsGetString(nullptr, nullptr, "-mesh_file", mesh_file, 256, &flag);

    mfem::Mesh serial_mesh;
    if (!flag) {
      serial_mesh = mfem::Mesh::MakeCartesian2D(32, 32, mfem::Element::Type::TRIANGLE);
    } else {
      serial_mesh = mfem::Mesh::LoadFromFile(mesh_file);
    }
    PetscInt refine = 1;
    PetscOptionsGetInt(nullptr, nullptr, "-refine", &refine, nullptr);
    for (PetscInt i = 0; i < refine; ++i) serial_mesh.UniformRefinement();

    mesh = new mfem::ParMesh(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    PetscInt maxGlobalElements = 10000;
    PetscOptionsGetInt(nullptr, nullptr, "-max_global_elements", &maxGlobalElements, nullptr);
    while (mesh->GetGlobalNE() < maxGlobalElements) mesh->UniformRefinement();

    PetscInt order = 1;
    PetscOptionsGetInt(nullptr, nullptr, "-order", &order, nullptr);
    fec = new mfem::H1_FECollection(order, mesh->Dimension());

    fespace = std::make_unique<mfem::ParFiniteElementSpace>(mesh, fec);

    // Pure Dirichlet boundary conditions
    mfem::Array<int> ess_bdr(fespace->GetParMesh()->bdr_attributes.Max());
    ess_bdr = 1;
    // for (int i = 0; i < ess_bdr.Size() / 2; ++i) ess_bdr[i] = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    PetscScalar kappa = 1;
    PetscOptionsGetReal(nullptr, nullptr, "-kappa", &kappa, nullptr);
    mfem::ConstantCoefficient kappa2(kappa * kappa);

    mfem::ParBilinearForm a(fespace.get());
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
    a.AddDomainIntegrator(new mfem::MassIntegrator(kappa2));
    a.SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
    a.SetOperatorType(mfem::Operator::PETSC_MATAIJ);
    a.Assemble(0);

    mfem::ParLinearForm       b(fespace.get());
    mfem::ConstantCoefficient one;
    b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
    b.Assemble();

    mfem::ParGridFunction u(fespace.get());
    u = 0.0;

    mfem::Vector X;
    mfem::Vector F;
    a.FormLinearSystem(ess_tdof_list, u, b, A, X, F);
    B = std::make_unique<mfem::PetscParVector>(MPI_COMM_WORLD, F, true);

    MatSetOption(A, MAT_SPD, PETSC_TRUE);

    // Create measurement vector
    AssembleMassMatrix();

    meas_vec = std::make_unique<mfem::PetscParVector>(A);
    mfem::ParGridFunction     proj(fespace.get());
    mfem::FunctionCoefficient obs([&](const mfem::Vector &coord) {
      if (coord.Norml2() < 0.3) return 1 / VolumeOfSphere(0.3);
      else return 0.;
    });

    proj.ProjectCoefficient(obs);
    M.Mult(proj, *meas_vec);
  }

  mfem::PetscParMatrix       &GetPrecisionMatrix() { return A; }
  const mfem::PetscParMatrix &GetPrecisionMatrix() const { return A; }

  mfem::PetscParVector       &GetRHSVector() { return *B; }
  const mfem::PetscParVector &GetRHSVector() const { return *B; }

  mfem::PetscParVector       &GetMeasurementVector() { return *meas_vec; }
  const mfem::PetscParVector &GetMeasurementVector() const { return *meas_vec; }

  mfem::ParFiniteElementSpace       &GetFiniteElementSpace() { return *fespace; }
  const mfem::ParFiniteElementSpace &GetFiniteElementSpace() const { return *fespace; }

  void AddObservations(double sigma2, const std::vector<mfem::Vector> &coords, std::vector<double> radii, std::vector<double> obsvals)
  {
    mfem::PetscParVector g(A);
    PetscInt             lsize, gsize;
    PetscCallVoid(VecGetLocalSize(g, &lsize));
    PetscCallVoid(VecGetSize(g, &gsize));

    Mat _BM;
    PetscCallVoid(MatCreateDense(A.GetComm(), lsize, PETSC_DECIDE, gsize, radii.size(), nullptr, &_BM));
    mfem::PetscParMatrix BM(_BM, false);

    mfem::PetscParVector S(BM), f(BM, true), y(S);
    S = 1. / sigma2;

    mfem::PetscParVector  meas(A);
    mfem::ParGridFunction proj(fespace.get());
    for (std::size_t i = 0; i < coords.size(); ++i) {
      meas = 0;
      proj = 0;

      mfem::FunctionCoefficient obs([&](const mfem::Vector &coord) {
        if (coord.DistanceTo(coords[i]) < radii[i]) return 1 / VolumeOfSphere(radii[i]);
        else return 0.;
      });

      proj.ProjectCoefficient(obs);
      M.Mult(proj, meas);

      Vec col;
      PetscCallVoid(MatDenseGetColumnVec(BM, i, &col));
      PetscCallVoid(VecCopy(meas, col));
      PetscCallVoid(MatDenseRestoreColumnVec(BM, i, &col));
      y[i] = obsvals[i];
    }

    PetscCallVoid(VecPointwiseMult(y, y, S));
    BM.Mult(y, f);

    Mat ALRC;
    Mat Ao = A.ReleaseMat(false);
    PetscCallVoid(MatCreateLRC(Ao, BM, S, nullptr, &ALRC));
    PetscCallVoid(PetscObjectDereference((PetscObject)Ao));

    A.SetMat(ALRC);
    PetscCallVoid(PetscObjectDereference((PetscObject)ALRC));
    *B = f;
  }

  ~Problem()
  {
    delete mesh;
    delete fec;
  }

private:
  void AssembleMassMatrix()
  {
    mfem::ParBilinearForm a(fespace.get());
    a.AddDomainIntegrator(new mfem::MassIntegrator);
    a.SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
    a.SetOperatorType(mfem::Operator::PETSC_MATAIJ);
    a.Assemble(0);
    a.FormSystemMatrix(ess_tdof_list, M);
  }

  double VolumeOfSphere(double radius)
  {
    switch (fespace->GetParMesh()->Dimension()) {
    case 2:
      return PETSC_PI * radius * radius;
    case 3:
      return 4 * PETSC_PI * radius * radius * radius;
    default:
      return 0;
    }
  }

  mfem::Array<int>                             ess_tdof_list;
  mfem::PetscParMatrix                         A;
  mfem::PetscParMatrix                         M;
  std::unique_ptr<mfem::PetscParVector>        B;
  std::unique_ptr<mfem::PetscParVector>        meas_vec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fespace;

  mfem::ParMesh         *mesh;
  mfem::H1_FECollection *fec;
};

inline PetscErrorCode CreateMatrixMFEM(Parameters params, Mat *A)
{
  std::unique_ptr<Problem> problem;
  (void)params;

  PetscFunctionBeginUser;
  mfem::Hypre::Init();
  problem = std::make_unique<Problem>();
  *A      = problem->GetPrecisionMatrix().ReleaseMat(false);
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
