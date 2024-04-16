#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/multicolor_gibbs.hh"

#include <mfem.hpp>

#include <mfem/fem/pgridfunc.hpp>
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

    for (int l = 0; l < fespaces.GetNumLevels(); ++l) {
      forms.emplace_back(std::make_unique<mfem::ParBilinearForm>(&fespaces.GetFESpaceAtLevel(l)));

      forms[l]->AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
      forms[l]->AddDomainIntegrator(new mfem::MassIntegrator(kappa2));
      forms[l]->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
      forms[l]->Assemble(0);

      forms[l]->SetOperatorType(mfem::Operator::PETSC_MATAIJ);

      mfem::Array<int> essTdofs;
      fespaces.GetFESpaceAtLevel(l).GetEssentialTrueDofs(essBdr, essTdofs);

      auto *a = new mfem::PetscParMatrix;
      forms[l]->FormSystemMatrix(essTdofs, *a);

      this->ops[l] = std::make_shared<parmgmc::LinearOperator>(*a);

      MatSetOption(*a, MAT_SPD, PETSC_TRUE);

      // this->ops[l]->color_matrix();
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
  std::vector<std::unique_ptr<mfem::ParBilinearForm>> forms;
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
  params.cycleType = parmgmc::MGMCCycleType::W;
  params.smoothingType = parmgmc::MGMCSmoothingType::ForwardBackward;
  params.coarseSamplerType = parmgmc::MGMCCoarseSamplerType::Cholesky;

  ShiftedLaplaceMGMC sampler(fespaces, essBdr, engine, params, kappainv);

  mfem::PetscParVector sample(&fespaces.GetFinestFESpace());
  mfem::PetscParVector rhs(sample);

  mfem::PetscParVector tgtMean(sample);
  // tgt_mean.Randomize();
  tgtMean = 5;

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

  {
    // auto &fespace = fespaces.GetFESpaceAtLevel(2);
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
