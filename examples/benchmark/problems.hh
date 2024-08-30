#pragma once

#include "params.hh"
#include "parmgmc/ms.h"
#include "parmgmc/obs.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscstring.h>
#include <petscsys.h>
#include <petscsystypes.h>

#if __has_include(<mfem.hpp>)
  #define PARMGMC_HAVE_MFEM
  #include <mfem.hpp>
#endif

struct MeasCtx {
  PetscScalar *centre, radius; // used when qoi is average over sphere
  PetscScalar *start, *end;    // used when qoi is average over rect/cuboid
  PetscScalar  vol;
};

inline PetscErrorCode f_sphere(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
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

inline PetscErrorCode f_rect(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)time;
  (void)Nc;
  MeasCtx  *octx   = (MeasCtx *)ctx;
  PetscBool inside = PETSC_TRUE;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < dim; ++i) {
    if (x[i] < octx->start[i] || x[i] > octx->end[i]) {
      inside = PETSC_FALSE;
      break;
    }
  }
  if (inside) *u = 1 / octx->vol;
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

inline PetscErrorCode VolumeOfRect(DM dm, PetscScalar *start, PetscScalar *end, PetscScalar *v)
{
  PetscInt cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(cdim == 2 || cdim == 3, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only dim=2 and dim=3 supported");
  *v = 1;
  for (PetscInt i = 0; i < cdim; ++i) *v *= end[i] - start[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode CreateMeshFromFilename(MPI_Comm comm, const char *filename, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateGmshFromFile(comm, filename, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, nullptr, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode CreateMatrixPetsc(Parameters params, Mat *A, Vec *meas_vec, DM *dm, Vec *rhs)
{
  MS        ms;
  char      filename[512];
  PetscBool flag;

  PetscFunctionBeginUser;
  PetscCall(MSCreate(MPI_COMM_WORLD, &ms));
  PetscCall(MSSetAssemblyOnly(ms, PETSC_TRUE));
  PetscCall(MSSetFromOptions(ms));

  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-mesh_file", filename, 512, &flag));
  if (flag) {
    DM mdm;

    PetscCall(CreateMeshFromFilename(MPI_COMM_WORLD, filename, &mdm));
    PetscCall(MSSetDM(ms, mdm));
  }
  PetscCall(MSSetUp(ms));
  PetscCall(MSGetPrecisionMatrix(ms, A));
  PetscCall(MSGetDM(ms, dm));

  if (params->with_lr) {
    PetscInt   nobs, cdim, nobs_given;
    PetscReal *obs_coords, *obs_radii, *obs_values, obs_sigma2 = 1e-4;

    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-nobs", &nobs, nullptr));
    PetscCall(DMGetCoordinateDim(*dm, &cdim));

    nobs_given = nobs * cdim;
    PetscCall(PetscMalloc1(nobs_given, &obs_coords));
    PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-obs_coords", obs_coords, &nobs_given, nullptr));
    PetscCheck(nobs_given == nobs * cdim, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Wrong number of observation coordinates provided, expected %d got %d", nobs * cdim, nobs_given);

    PetscCall(PetscMalloc1(nobs, &obs_radii));
    nobs_given = nobs;
    PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-obs_radii", obs_radii, &nobs_given, nullptr));
    PetscCheck(nobs_given == 1 || nobs_given == nobs, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Wrong number of observation radii provided, expected either 1 or %d got %d", nobs, nobs_given);
    if (nobs_given == 1)
      for (PetscInt i = 1; i < nobs; ++i) obs_radii[i] = obs_radii[0]; // If only one radius provided, use that for all observations

    PetscCall(PetscMalloc1(nobs, &obs_values));
    nobs_given = nobs;
    PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-obs_values", obs_values, &nobs_given, nullptr));
    PetscCheck(nobs_given == 1 || nobs_given == nobs, MPI_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Wrong number of observation values provided, expected either 1 or %d got %d", nobs, nobs_given);
    if (nobs_given == 1)
      for (PetscInt i = 1; i < nobs; ++i) obs_values[i] = obs_values[0]; // If only one value provided, use that for all observations

    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-obs_sigma2", &obs_sigma2, nullptr));

    Mat A2, B;
    Vec S;
    PetscCall(MakeObservationMats(*dm, nobs, obs_sigma2, obs_coords, obs_radii, obs_values, &B, &S, rhs));
    PetscCall(MatCreateLRC(*A, B, S, nullptr, &A2));
    PetscCall(MatDestroy(&B));
    PetscCall(VecDestroy(&S));
    PetscCall(PetscFree(obs_coords));
    PetscCall(PetscFree(obs_radii));
    PetscCall(PetscFree(obs_values));

    *A = A2;
  } else {
    PetscCall(PetscObjectReference((PetscObject)(*A)));
    PetscCall(DMCreateGlobalVector(*dm, rhs));
  }
  PetscCall(PetscObjectReference((PetscObject)(*dm))); // Make sure MSDestroy doesn't destroy the DM because we're returning it
  PetscCall(MSDestroy(&ms));

  // Create measurement vector
  Mat       M;
  Vec       u;
  MeasCtx   ctx;
  PetscInt  dim, got_dim;
  void     *mctx         = &ctx;
  char      qoi_type[64] = "sphere";
  PetscBool valid_type;

  PetscCall(DMCreateGlobalVector(*dm, meas_vec));
  PetscCall(DMCreateMassMatrix(*dm, *dm, &M));
  PetscCall(DMGetGlobalVector(*dm, &u));

  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-qoi_type", qoi_type, 64, nullptr));
  PetscCall(PetscStrcmpAny(qoi_type, &valid_type, "sphere", "rect", ""));
  PetscCheck(valid_type, MPI_COMM_WORLD, PETSC_ERR_SUP, "-qoi_type must be sphere or rect");

  PetscCall(PetscStrcmp(qoi_type, "sphere", &flag));
  if (flag) {
    PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {f_sphere};

    PetscCall(DMGetCoordinateDim(*dm, &dim));
    PetscCall(PetscCalloc1(dim, &ctx.centre));
    got_dim = dim;
    PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_centre", ctx.centre, &got_dim, nullptr));
    PetscCheck(got_dim == 0 or got_dim == dim, MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed, expected %d", dim);
    ctx.radius = 1;
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-qoi_radius", &ctx.radius, nullptr));
    PetscCall(VolumeOfSphere(*dm, ctx.radius, &ctx.vol));
    PetscCall(DMProjectFunction(*dm, 0, funcs, &mctx, INSERT_VALUES, u));
    PetscCall(MatMult(M, u, *meas_vec));
  } else {
    PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {f_rect};

    PetscCall(DMGetCoordinateDim(*dm, &dim));
    PetscCall(PetscCalloc1(dim, &ctx.start));
    PetscCall(PetscCalloc1(dim, &ctx.end));
    for (PetscInt i = 0; i < dim; ++i) ctx.end[i] = 1;
    got_dim = dim;
    PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_start", ctx.start, &got_dim, nullptr));
    PetscCheck(got_dim == 0 or got_dim == dim, MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed for start, expected %d", dim);
    got_dim = dim;
    PetscCall(PetscOptionsGetRealArray(nullptr, nullptr, "-qoi_end", ctx.end, &got_dim, nullptr));
    PetscCheck(got_dim == 0 or got_dim == dim, MPI_COMM_WORLD, PETSC_ERR_SUP, "Incorrect number of points passed for end, expected %d", dim);

    PetscCall(VolumeOfRect(*dm, ctx.start, ctx.end, &ctx.vol));
    PetscCall(DMProjectFunction(*dm, 0, funcs, &mctx, INSERT_VALUES, u));
    PetscCall(MatMult(M, u, *meas_vec));
  }
  // {
  //   PetscViewer viewer;
  //   char        filename[512] = "measurement_vec.vtu";

  //   PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", filename, 512, NULL));
  //   PetscCall(PetscViewerVTKOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));

  //   PetscCall(PetscObjectSetName((PetscObject)(u), "u"));
  //   PetscCall(VecView(u, viewer));

  //   PetscCall(PetscObjectSetName((PetscObject)(*meas_vec), "meas_vec"));
  //   PetscCall(VecView(*meas_vec, viewer));

  //   PetscCall(PetscViewerDestroy(&viewer));
  // }
  PetscCall(DMRestoreGlobalVector(*dm, &u));
  PetscCall(MatDestroy(&M));
  if (flag) {
    PetscCall(PetscFree(ctx.centre));
  } else {
    PetscCall(PetscFree(ctx.start));
    PetscCall(PetscFree(ctx.end));
  }
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
      PetscInt faces[2];

      faces[0] = 4;
      PetscOptionsGetInt(nullptr, nullptr, "-box_faces", &faces[0], nullptr);
      faces[1]    = faces[0];
      serial_mesh = mfem::Mesh::MakeCartesian2D(faces[0], faces[1], mfem::Element::Type::TRIANGLE);
    } else {
      serial_mesh = mfem::Mesh::LoadFromFile(mesh_file);
    }
    PetscInt refine = 0;
    PetscOptionsGetInt(nullptr, nullptr, "-dm_refine", &refine, nullptr);
    for (PetscInt i = 0; i < refine; ++i) serial_mesh.UniformRefinement();

    mesh = new mfem::ParMesh(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    PetscInt parRefine = 0;
    PetscOptionsGetInt(nullptr, nullptr, "-dm_par_refine", &parRefine, nullptr);
    for (PetscInt i = 0; i < parRefine; ++i) mesh->UniformRefinement();

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
    PetscOptionsGetReal(nullptr, nullptr, "-matern_kappa", &kappa, nullptr);
    mfem::ConstantCoefficient kappa2(kappa * kappa);

    mfem::ParBilinearForm a(fespace.get());
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
    a.AddDomainIntegrator(new mfem::MassIntegrator(kappa2));
    a.SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
    a.SetOperatorType(mfem::Operator::PETSC_MATAIJ);
    a.Assemble(0);

    mfem::ParLinearForm       b(fespace.get());
    mfem::ConstantCoefficient zero(0);
    b.AddDomainIntegrator(new mfem::DomainLFIntegrator(zero));
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
      double radius = 0.7;
      if (coord.Norml2() < radius) return 1 / VolumeOfSphere(radius);
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

inline PetscErrorCode CreateMatrixMFEM(Parameters params, Vec *meas_vec, Mat *A, Vec *rhs)
{
  std::unique_ptr<Problem> problem;
  (void)params;

  PetscFunctionBeginUser;
  mfem::Hypre::Init();
  problem   = std::make_unique<Problem>();
  *A        = problem->GetPrecisionMatrix().ReleaseMat(false);
  *meas_vec = problem->GetMeasurementVector();
  *rhs      = problem->GetRHSVector();
  PetscCall(PetscObjectReference((PetscObject)(*meas_vec)));
  PetscCall(PetscObjectReference((PetscObject)(*rhs)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
