#include "mat.hh"
#include "qoi.hh"

#include "parmgmc/common/helpers.hh"
#include "parmgmc/common/petsc_helper.hh"
#include "parmgmc/dm_hierarchy.hh"
#include "parmgmc/linear_operator.hh"
#include "parmgmc/samplers/mgmc.hh"
#include "parmgmc/samplers/sample_chain.hh"

#if PETSC_HAVE_MKL_CPARDISO
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
inline PetscErrorCode iact(const std::string &name, Chain &chain,
                           Vec sample_rhs) {
  PetscFunctionBeginUser;

  Timer timer;

  Vec initial_sample;
  PetscCall(VecDuplicate(sample_rhs, &initial_sample));

  for (std::size_t n = 0; n < chain.get_n_chains(); ++n) {
    PetscCall(VecSet(initial_sample, (n + 1) * 10));
    chain.set_sample(initial_sample, n);
  }
  PetscCall(VecDestroy(&initial_sample));

  PetscInt n_burnin = 20;
  PetscOptionsGetInt(nullptr, nullptr, "-n_burnin", &n_burnin, nullptr);

  PetscPrintf(MPI_COMM_WORLD, "Starting burnin...");
  timer.reset();
  PetscCall(chain.sample(sample_rhs, n_burnin));
  PetscPrintf(MPI_COMM_WORLD, "Done. Took %f seconds.\n", timer.elapsed());
  chain.reset();

  PetscInt n_samples = 20;
  PetscOptionsGetInt(nullptr, nullptr, "-n_samples", &n_samples, nullptr);

  PetscPrintf(MPI_COMM_WORLD, "Starting sampling...");
  timer.reset();
  PetscCall(chain.sample(sample_rhs, n_samples));
  auto elapsed = timer.elapsed();

  auto chain_iact = chain.integrated_autocorr_time();

  PetscPrintf(MPI_COMM_WORLD,
              "Done. Took %f seconds, %f seconds per sample, %f seconds per "
              "independent sample.\n",
              elapsed,
              elapsed / n_samples,
              chain_iact * elapsed / n_samples);

  PetscCall(PetscPrintf(MPI_COMM_WORLD,
                        "%s IACT: %zu (has %sconverged, R = %f)\n",
                        name.c_str(),
                        chain_iact,
                        chain.converged() ? "" : "not ",
                        chain.gelman_rubin()));

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
  std::shared_ptr<DMHierarchy> dm_hierarchy;
  {
    const PetscInt dof_per_node = 1;
    const PetscInt stencil_width = 1;

    int n_vertices = 5;
    PetscOptionsGetInt(nullptr, nullptr, "-n_vertices", &n_vertices, nullptr);

    Coordinate lower_left{0, 0};
    Coordinate upper_right{1, 1};

    DM dm;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE,
                           DM_BOUNDARY_NONE,
                           DMDA_STENCIL_STAR,
                           n_vertices,
                           n_vertices,
                           PETSC_DECIDE,
                           PETSC_DECIDE,
                           dof_per_node,
                           stencil_width,
                           nullptr,
                           nullptr,
                           &dm));

    PetscCall(DMSetUp(dm));
    PetscCall(DMDASetUniformCoordinates(
        dm, lower_left.x, upper_right.x, lower_left.y, upper_right.y, 0, 0));

    PetscInt n_levels = 5;
    PetscOptionsGetInt(nullptr, nullptr, "-n_levels", &n_levels, nullptr);

    dm_hierarchy = std::make_shared<DMHierarchy>(dm, n_levels);
    // PetscCall(dm_hierarchy->print_info());
  }

  int n_chains = 8;
  PetscOptionsGetInt(nullptr, nullptr, "-n_chains", &n_chains, nullptr);

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
  Vec sample_rhs;
  PetscCall(DMCreateGlobalVector(dm_hierarchy->get_fine(), &sample_rhs));
  PetscCall(fill_vec_rand(sample_rhs, engine));

  NormQOI qoi;

  PetscInt n_smooth = 2;
  PetscOptionsGetInt(nullptr, nullptr, "-n_smooth", &n_smooth, nullptr);

  MGMCParameters params;
  params.n_smooth = n_smooth;
  params.cycle_type = MGMCCycleType::V;
  params.smoothing_type = MGMCSmoothingType::Symmetric;
#if PETSC_HAVE_MKL_CPARDISO
  params.coarse_sampler_type = MGMCCoarseSamplerType::Cholesky;
#else
  params.coarse_sampler_type = MGMCCoarseSamplerType::Standard;
#endif

  // Setup Multigrid sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up multigrid sampler...\n"));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dm_hierarchy->get_fine(), &mat));
    auto linear_operator = std::make_shared<LinearOperator>(mat);

    using Chain = SampleChain<MultigridSampler<pcg32>, NormQOI>;
    Chain chain(qoi,
                n_chains,
                sample_rhs,
                linear_operator,
                dm_hierarchy,
                &engine,
                params);

    PetscCall(iact("MGMC", chain, sample_rhs));
  }

  // Setup Gibbs sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up Gibbs sampler...\n"));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dm_hierarchy->get_fine(), &mat));
    auto linear_operator = std::make_shared<LinearOperator>(mat);
    // linear_operator->color_matrix(dm_hierarchy->get_fine());

    PetscReal omega = 1.; // SOR parameter
    PetscOptionsGetReal(nullptr, nullptr, "-omega", &omega, nullptr);

    using Chain = SampleChain<MulticolorGibbsSampler<pcg32>, NormQOI>;
    Chain chain(qoi,
                n_chains,
                sample_rhs,
                linear_operator,
                &engine,
                omega,
                GibbsSweepType::Symmetric);

    PetscCall(iact("Gibbs", chain, sample_rhs));
  }

#if PETSC_HAVE_MKL_CPARDISO
  // Setup Cholesky sampler
  {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Setting up Cholesky sampler...\n"));

    // Setup fine operator
    Mat mat;
    PetscCall(assemble(dm_hierarchy->get_fine(), &mat));
    auto linear_operator = std::make_shared<LinearOperator>(mat);

    using Chain = SampleChain<CholeskySampler<pcg32>, NormQOI>;
    Chain chain(qoi, n_chains, sample_rhs, linear_operator, &engine);

    PetscCall(iact("Cholesky", chain, sample_rhs));
  }
#endif

  PetscCall(VecDestroy(&sample_rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
