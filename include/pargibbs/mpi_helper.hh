#pragma once

#include <cstdlib>
#include <stdexcept>
#include <utility>

#if USE_MPI
#include <mpi.h>
#else
#include "FakeMPI/mpi.h"
#endif

namespace pargibbs {
[[maybe_unused]] static bool mpi_is_initialised = false;

struct mpi_helper {
  mpi_helper(int *argc = nullptr, char ***argv = nullptr) {
    MPI_Init(argc, argv);
    mpi_is_initialised = true;
  }

  ~mpi_helper() { MPI_Finalize(); }

  static std::pair<int, int> get_size_rank() {
    assert_initalised();
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return {size, rank};
  }

  static int get_rank() {
    assert_initalised();
    return get_size_rank().second;
  }
  static int get_size() {
    assert_initalised();
    return get_size_rank().first;
  }

  static int debug_rank() {
    assert_initalised();

    if (const char *env_rank = std::getenv("PARGIBBS_DEBUG_RANK"))
      return atoi(env_rank);
    else
      return 0;
  }

  static bool is_debug_rank() {
    assert_initalised();
    return get_rank() == debug_rank();
  }

private:
  static void assert_initalised() {
    if (!mpi_is_initialised)
      throw std::runtime_error(
          "Construct a pargibbs::mpi_helper object in main().");
  }
};
}; // namespace pargibbs
