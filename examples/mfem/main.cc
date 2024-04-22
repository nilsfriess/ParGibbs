#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <mfem.hpp>

#include <mpi.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <pcg_random.hpp>

#include <memory>
#include <random>
#include <string>

template <class Engine>
using MGMC = parmgmc::MultigridSampler<Engine, parmgmc::MulticolorGibbsSampler<Engine>>;

template <class Engine> class ShiftedLaplaceMGMC : public MGMC<Engine> {
public:
  ShiftedLaplaceMGMC(mfem::ParFiniteElementSpaceHierarchy &fespaces, const mfem::Array<int> &essBdr,
                     Engine &engine, const parmgmc::MGMCParameters &params, double kappainv)
      : MGMC<Engine>(params, fespaces.GetNumLevels(), engine), fespaces{fespaces} {
    this->ops.resize(fespaces.GetNumLevels());

    mfem::ConstantCoefficient one(1.0);
    mfem::ConstantCoefficient kappa2(1. / (kappainv * kappainv));

    fineForm = std::make_unique<mfem::ParBilinearForm>(&fespaces.GetFinestFESpace());
    fineForm->AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
    fineForm->AddDomainIntegrator(new mfem::MassIntegrator(kappa2));
    fineForm->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
    fineForm->Assemble(0);
    fineForm->Finalize(0);

    fineForm->SetOperatorType(mfem::Operator::PETSC_MATAIJ);
    mfem::Array<int> essTdofs;
    fespaces.GetFinestFESpace().GetEssentialTrueDofs(essBdr, essTdofs);

    auto *a = new mfem::PetscParMatrix;
    fineForm->FormSystemMatrix(essTdofs, *a);
    Mat pa = a->ReleaseMat(false);

    this->ops[fespaces.GetFinestLevelIndex()] = std::make_shared<parmgmc::LinearOperator>(pa, true);
    this->ops[fespaces.GetFinestLevelIndex()]->colorMatrix();

    for (int l = fespaces.GetFinestLevelIndex(); l > 0; --l) {
      mfem::OperatorHandle p(mfem::Operator::Hypre_ParCSR);
      fespaces.GetFESpaceAtLevel(l).GetTrueTransferOperator(fespaces.GetFESpaceAtLevel(l - 1), p);

      auto *hp = p.As<mfem::HypreParMatrix>();
      mfem::PetscParMatrix pp{hp};

      PetscObjectReference((PetscObject)this->ops[l]->getMat());

      mfem::PetscParMatrix fineMat(this->ops[l]->getMat());

      auto coarseMat = mfem::RAP(&fineMat, &pp)->ReleaseMat(false);
      this->ops[l - 1] = std::make_shared<parmgmc::LinearOperator>(coarseMat, true);
      this->ops[l - 1]->colorMatrix();
    }
  }

  PetscErrorCode restrict(std::size_t level, Vec residual, Vec rhs) override {
    mfem::PetscParVector r(residual, true);
    mfem::PetscParVector b(rhs, true);

    fespaces.GetProlongationAtLevel(level - 1)->MultTranspose(r, b);

    return PETSC_SUCCESS;
  }

  PetscErrorCode prolongateAdd(std::size_t level, Vec coarse, Vec fine) override {
    mfem::PetscParVector xc(coarse, true);
    mfem::PetscParVector xf(fine, true);

    fespaces.GetProlongationAtLevel(level - 1)->AddMult(xc, xf);

    return PETSC_SUCCESS;
  }

private:
  std::unique_ptr<mfem::ParBilinearForm> fineForm;
  mfem::ParFiniteElementSpaceHierarchy &fespaces;
};

int main(int argc, char *argv[]) {
  parmgmc::PetscHelper::init(argc, argv);

  std::string meshFile("../data/star.mesh");

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&meshFile, "-m", "--mesh", "Mesh file to use");

  int nLevels = 3;
  args.AddOption(&nLevels, "-l", "--levels", "Number of Multigrid levels");

  double kappainv = 1.;
  args.AddOption(&kappainv, "-k", "--kappainv", "Correlation length kappa");

  int nSamples = 10;
  args.AddOption(&nSamples, "-n", "--samples", "Number of samples");

  bool visualise = false;
  args.AddOption(&visualise, "-v", "--visualise", "-nv", "-no-visualise", "Save Paraview files");

  args.Parse();
  // if (!args.Good()) {
  //   args.PrintUsage(std::cout);
  //   return 1;
  // }

  mfem::Mesh mesh(meshFile, 1, 1);
  const auto dim = mesh.Dimension();

  /* Refine the serial mesh such that the refined mesh has at most 1000
   * elements. */
  int refLevels = std::floor(std::log(1000. / mesh.GetNE()) / std::log(2) / dim);
  for (int l = 0; l < refLevels; l++)
    mesh.UniformRefinement();

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  pmesh.UniformRefinement();
  pmesh.UniformRefinement();

  const int order = 1;
  mfem::H1_FECollection fec(order, dim);
  mfem::ParFiniteElementSpace coarseFespace(&pmesh, &fec);

  mfem::ParFiniteElementSpaceHierarchy fespaces(&pmesh, &coarseFespace, false, false);
  for (int l = 0; l < nLevels - 1; ++l)
    fespaces.AddUniformlyRefinedLevel();

  PetscPrintf(MPI_COMM_WORLD, "Number of FE unknowns: %d\n",
              fespaces.GetFinestFESpace().GlobalTrueVSize());

  mfem::Array<int> essBdr(pmesh.bdr_attributes.Max());
  if (pmesh.bdr_attributes.Size())
    essBdr = 1;

  pcg32 engine;
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seed;
    if (rank == 0) {
      seed = std::random_device{}();
      PetscOptionsGetInt(nullptr, nullptr, "-seed", &seed, nullptr);
    }

    // Send seed to all other processes
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    engine.seed(seed);
    engine.set_stream(rank);
  }

  parmgmc::MGMCParameters params;
  params.nSmooth = 2;
  params.cycleType = parmgmc::MGMCCycleType::V;
  params.smoothingType = parmgmc::MGMCSmoothingType::ForwardBackward;
  params.coarseSamplerType = parmgmc::MGMCCoarseSamplerType::Cholesky;

  ShiftedLaplaceMGMC sampler(fespaces, essBdr, engine, params, kappainv);

  mfem::PetscParVector sample(&fespaces.GetFinestFESpace());
  mfem::PetscParVector rhs(sample);

  mfem::PetscParVector tgtMean(sample);

  tgtMean = 0;
  sample = 0;

  MatMult(sampler.getOperator(fespaces.GetFinestLevelIndex()).getMat(), tgtMean, rhs);

  mfem::PetscParVector mean(sample);
  mean = 0;

  mfem::PetscParVector err(mean);

  const int nSaveSamples = 5;
  std::vector<mfem::ParGridFunction> saveSamples;

  for (int n = 0; n < nSamples; ++n) {
    sampler.sample(sample, rhs, 1);

    mean.Add(1. / nSamples, sample);

    VecWAXPY(err, -1, mean, tgtMean);

    if (nSamples - n <= nSaveSamples) {
      saveSamples.emplace_back(&fespaces.GetFinestFESpace());
      saveSamples.back().SetFromTrueDofs(sample);
    }

    PetscPrintf(MPI_COMM_WORLD, "%f\n", err.Normlinf());
  }

  if (visualise) {
    auto &fespace = fespaces.GetFinestFESpace();

    mfem::ParGridFunction xmean(&fespace);
    xmean.SetFromTrueDofs(mean);

    mfem::ParaViewDataCollection pd("Star", fespace.GetParMesh());
    pd.SetPrefixPath("ParaView");

    for (std::size_t i = 0; i < saveSamples.size(); ++i)
      pd.RegisterField("Sample " + std::to_string(i), &saveSamples[i]);

    pd.RegisterField("Mean", &xmean);
    pd.SetLevelsOfDetail(order);
    pd.SetDataFormat(mfem::VTKFormat::BINARY);
    pd.SetHighOrderOutput(true);
    pd.SetCycle(0);
    pd.SetTime(0.0);
    pd.Save();
  }

  return 0;
}
