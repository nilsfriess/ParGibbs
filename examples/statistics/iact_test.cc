#include "mat.hh"
#include "qoi.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/sample_chain.hh"

#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
#include "parmgmc/samplers/cholesky.hh"
#endif

#include <mpi.h>
#include <pcg_random.hpp>

#include <petscdm.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include <memory>
#include <random>

using namespace parmgmc;

template <class Chain>
inline PetscErrorCode iact(const std::string &name, Chain &chain, Vec sampleRhs) {
  PetscFunctionBeginUser;

  Timer timer;

  Vec initialSample;
  PetscCall(VecDuplicate(sampleRhs, &initialSample));

  for (std::size_t n = 0; n < chain.getNChains(); ++n) {
    PetscCall(VecSet(initialSample, (n + 1) * 10));
    chain.setSample(initialSample, n);
  }
  PetscCall(VecDestroy(&initialSample));

  PetscInt nBurnin = 20;
  PetscOptionsGetInt(nullptr, nullptr, "-n_burnin", &nBurnin, nullptr);

  PetscPrintf(MPI_COMM_WORLD, "Starting burnin...");
  timer.reset();
  PetscCall(chain.sample(sampleRhs, nBurnin));
  PetscPrintf(MPI_COMM_WORLD, "Done. Took %f seconds.\n", timer.elapsed());
  chain.reset();

  PetscInt nSamples = 20;
  PetscOptionsGetInt(nullptr, nullptr, "-n_samples", &nSamples, nullptr);

  PetscPrintf(MPI_COMM_WORLD, "Starting sampling...");
  timer.reset();
  PetscCall(chain.sample(sampleRhs, nSamples));
  auto elapsed = timer.elapsed();

  auto chainIACT = chain.integratedAutocorrTime();

  PetscPrintf(MPI_COMM_WORLD,
              "Done. Took %f seconds, %f seconds per sample, %f seconds per "
              "independent sample.\n",
              elapsed, elapsed / nSamples, chainIACT * elapsed / nSamples);

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "%s IACT: %zu (has %sconverged, R = %f)\n", name.c_str(),
                        chainIACT, chain.converged() ? "" : "not ", chain.gelmanRubin()));

  PetscFunctionReturn(PETSC_SUCCESS);
}

struct Coordinate {
  PetscReal x;
  PetscReal y;
};

int main(int argc, char *argv[]) {
  PetscHelper::init(argc, argv);

  PetscFunctionBeginUser;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup DM hierarchy
  std::shared_ptr<DMHierarchy> dmHierarchy;
  {
    const PetscInt dofPerNode = 1;
    const PetscInt stencilWidth = 1;

    int nVertices = 5;
    PetscOptionsGetInt(nullptr, nullptr, "-n_vertices", &nVertices, nullptr);

    Coordinate lowerLeft{0, 0};
    Coordinate upperRight{1, 1};

    DM dm;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           nVertices, nVertices, PETSC_DECIDE, PETSC_DECIDE, dofPerNode,
                           stencilWidth, nullptr, nullptr, &dm));

    PetscCall(DMSetUp(dm));
    PetscCall(DMDASetUniformCoordinates(dm, lowerLeft.x, upperRight.x, lowerLeft.y,
                                        upperRight.y, 0, 0));

    PetscInt nLevels = 5;
    PetscOptionsGetInt(nullptr, nullptr, "-n_levels", &nLevels, nullptr);

    dmHierarchy = std::make_shared<DMHierarchy>(dm, nLevels);
    // PetscCall(dm_hierarchy->print_info());
  }

  int nChains = 8;
  PetscOptionsGetInt(nullptr, nullptr, "-n_chains", &nChains, nullptr);

  // Setup random number generator
  pcg32 engine;
  {
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

  // RHS used in samplers
  Vec sampleRhs;
  PetscCall(DMCreateGlobalVector(dmHierarchy->getFine(), &sampleRhs));
  PetscCall(fillVecRand(sampleRhs, engine));

  NormQOI qoi;

  PetscInt nSmooth = 2;
  PetscOptionsGetInt(nullptr, nullptr, "-n_smooth", &nSmooth, nullptr);

  MGMCParameters params;
  params.nSmooth = nSmooth;
  params.cycleType = MGMCCycleType::V;
  params.smoothingType = MGMCSmoothingType::ForwardBackward;
#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
  params.coarseSamplerType = MGMCCoarseSamplerType::Cholesky;
#else
  params.coarse_sampler_type = MGMCCoarseSamplerType::Standard;
#endif

  // Setup Multigrid sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up multigrid sampler...\n"));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dmHierarchy->getFine(), &mat));
    auto linearOperator = std::make_shared<LinearOperator>(mat);

    using Chain = SampleChain<MultigridSampler<pcg32>, NormQOI>;
    Chain chain(qoi, nChains, sampleRhs, linearOperator, dmHierarchy, &engine, params);

    PetscCall(iact("MGMC", chain, sampleRhs));
  }

  // Setup Gibbs sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up Gibbs sampler...\n"));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dmHierarchy->getFine(), &mat));
    auto linearOperator = std::make_shared<LinearOperator>(mat);
    // linear_operator->color_matrix(dm_hierarchy->get_fine());

    PetscReal omega = 1.; // SOR parameter
    PetscOptionsGetReal(nullptr, nullptr, "-omega", &omega, nullptr);

    using Chain = SampleChain<MulticolorGibbsSampler<pcg32>, NormQOI>;
    Chain chain(qoi, nChains, sampleRhs, linearOperator, &engine, omega,
                GibbsSweepType::Symmetric);

    PetscCall(iact("Gibbs", chain, sampleRhs));
  }

#if PETSC_HAVE_MKL_CPARDISO && PETSC_HAVE_MKL_PARDISO
  // Setup Cholesky sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up Cholesky sampler...\n"));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dmHierarchy->getFine(), &mat));
    auto linearOperator = std::make_shared<LinearOperator>(mat);

    using Chain = SampleChain<CholeskySampler<pcg32>, NormQOI>;
    Chain chain(qoi, nChains, sampleRhs, linearOperator, &engine);

    PetscCall(iact("Cholesky", chain, sampleRhs));
  }
#endif

  PetscCall(VecDestroy(&sampleRhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
