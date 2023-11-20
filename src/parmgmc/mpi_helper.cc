#include <cstdlib>
#include <stdexcept>
#include <utility>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

#include "parmgmc/mpi_helper.hh"

namespace parmgmc {
static bool mpi_is_initialised = false;

int mpi_helper::_debug_rank = 0;

mpi_helper::mpi_helper(int *argc, char ***argv) {
  MPI_Init(argc, argv);
  mpi_is_initialised = true;

  if (const char *env_rank = std::getenv("PARMGMC_DEBUG_RANK"))
    _debug_rank = atoi(env_rank);
  else
    _debug_rank = 0;
}

mpi_helper::~mpi_helper() { MPI_Finalize(); }

std::pair<int, int> mpi_helper::get_size_rank() {
  assert_initalised();
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return {size, rank};
}

int mpi_helper::get_rank() {
  assert_initalised();
  return get_size_rank().second;
}
int mpi_helper::get_size() {
  assert_initalised();
  return get_size_rank().first;
}

int mpi_helper::debug_rank() {
  assert_initalised();
  return _debug_rank;
}

bool mpi_helper::is_debug_rank() {
  assert_initalised();
  return get_rank() == debug_rank();
}

void mpi_helper::assert_initalised() {
  if (!mpi_is_initialised)
    throw std::runtime_error(
        "Construct a parmgmc::mpi_helper object in main().");
}

} // namespace parmgmc
