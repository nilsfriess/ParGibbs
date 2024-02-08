#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/hogwild.hh"
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

#include <iostream>
#include <memory>
#include <random>
#include <string>

template <class Engine>
using MGMC =
    parmgmc::MultigridSampler<Engine, parmgmc::MulticolorGibbsSampler<Engine>>;

template <class Engine> class ShiftedLaplaceMGMC : public MGMC<Engine> {
public:
  ShiftedLaplaceMGMC(mfem::ParFiniteElementSpaceHierarchy &fespaces,
                     const mfem::Array<int> &ess_bdr, Engine *engine,
                     const parmgmc::MGMCParameters &params, double kappainv)
      : MGMC<Engine>(params, fespaces.GetNumLevels(), engine),
        fespaces{fespaces} {
    this->ops.resize(fespaces.GetNumLevels());

    mfem::ConstantCoefficient one(1.0);
    mfem::ConstantCoefficient kappa2(1. / (kappainv * kappainv));

    for (int l = 0; l < fespaces.GetNumLevels(); ++l) {
      forms.emplace_back(std::make_unique<mfem::ParBilinearForm>(
          &fespaces.GetFESpaceAtLevel(l)));

      forms[l]->AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
      forms[l]->AddDomainIntegrator(new mfem::MassIntegrator(kappa2));
      forms[l]->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
      forms[l]->Assemble(0);

      forms[l]->SetOperatorType(mfem::Operator::PETSC_MATAIJ);

      mfem::Array<int> ess_tdofs;
      fespaces.GetFESpaceAtLevel(l).GetEssentialTrueDofs(ess_bdr, ess_tdofs);

      auto *A = new mfem::PetscParMatrix;
      forms[l]->FormSystemMatrix(ess_tdofs, *A);
      this->ops[l] = std::make_shared<parmgmc::LinearOperator>(*A, false);
      this->ops[l]->color_matrix();
    }
  }

  PetscErrorCode restrict(std::size_t level, Vec residual, Vec rhs) override {
    mfem::PetscParVector r(residual, true);
    mfem::PetscParVector b(rhs, true);

    fespaces.GetProlongationAtLevel(level - 1)->MultTranspose(r, b);

    return PETSC_SUCCESS;
  }

  PetscErrorCode prolongate_add(std::size_t level, Vec coarse,
                                Vec fine) override {
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

  std::string mesh_file("../data/star.mesh");

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use");

  int n_levels = 3;
  args.AddOption(&n_levels, "-l", "--levels", "Number of Multigrid levels");

  double kappainv = 1.;
  args.AddOption(&kappainv, "-k", "--kappainv", "Correlation length kappa");

  int n_samples = 10;
  args.AddOption(&n_samples, "-n", "--samples", "Number of samples");

  args.Parse();
  // if (!args.Good()) {
  //   args.PrintUsage(std::cout);
  //   return 1;
  // }

  mfem::Mesh mesh(mesh_file, 1, 1);
  const auto dim = mesh.Dimension();

  /* Refine the serial mesh such that the refined mesh has at most 1000
   * elements. */
  int ref_levels =
      std::floor(std::log(1000. / mesh.GetNE()) / std::log(2) / dim);
  for (int l = 0; l < ref_levels; l++)
    mesh.UniformRefinement();

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  pmesh.UniformRefinement();
  pmesh.UniformRefinement();

  const int order = 1;
  mfem::H1_FECollection fec(order, dim);
  mfem::ParFiniteElementSpace coarse_fespace(&pmesh, &fec);

  mfem::ParFiniteElementSpaceHierarchy fespaces(
      &pmesh, &coarse_fespace, false, false);
  for (int l = 0; l < n_levels - 1; ++l)
    fespaces.AddUniformlyRefinedLevel();

  PetscPrintf(MPI_COMM_WORLD,
              "Number of FE unknowns: %d\n",
              fespaces.GetFinestFESpace().GlobalTrueVSize());

  mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
  if (pmesh.bdr_attributes.Size())
    ess_bdr = 1;

  pcg32 engine;
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seed;
    if (rank == 0) {
      seed = std::random_device{}();
      PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL);
    }

    // Send seed to all other processes
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    engine.seed(seed);
    engine.set_stream(rank);
  }

  parmgmc::MGMCParameters params;
  params.n_smooth = 1;
  params.cycle_type = parmgmc::MGMCCycleType::V;
  params.smoothing_type = parmgmc::MGMCSmoothingType::Symmetric;

  ShiftedLaplaceMGMC sampler(fespaces, ess_bdr, &engine, params, kappainv);

  mfem::PetscParVector sample(&fespaces.GetFinestFESpace());
  mfem::PetscParVector rhs(sample);

  mfem::PetscParVector tgt_mean(sample);
  // tgt_mean.Randomize();
  tgt_mean = 0;

  sample = 0;

  MatMult(sampler.get_operator(fespaces.GetFinestLevelIndex())->get_mat(),
          tgt_mean,
          rhs);

  mfem::PetscParVector mean(sample);
  mean = 0;

  mfem::PetscParVector err(mean);

  const int n_save_samples = 5;
  std::vector<mfem::ParGridFunction> save_samples;

  for (int n = 0; n < n_samples; ++n) {
    sampler.sample(sample, rhs, 1);

    mean.Add(1. / n_samples, sample);

    VecWAXPY(err, -1, mean, tgt_mean);

    if (n_samples - n <= n_save_samples) {
      save_samples.emplace_back(&fespaces.GetFinestFESpace());
      save_samples.back().SetFromTrueDofs(sample);
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

    for (std::size_t i = 0; i < save_samples.size(); ++i)
      pd.RegisterField("Sample " + std::to_string(i), &save_samples[i]);

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
