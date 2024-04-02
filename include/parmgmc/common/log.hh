#pragma once

#include <cstdlib>
#include <iostream>
#include <mpi.h>

#define PARMGMC_DEBUG_LEVEL_NONE 0
#define PARMGMC_DEBUG_LEVEL_INFO 1
#define PARMGMC_DEBUG_LEVEL_VERBOSE 2

#ifndef PARMGMC_DEBUG_LEVEL
#define PARMGMC_DEBUG_LEVEL PARMGMC_DEBUG_LEVEL_INFO
#endif

namespace parmgmc {
class logger_stream {
public:
  static logger_stream &get() {
    static logger_stream log;
    return log;
  }

  [[nodiscard]] std::ostream &get_stream() const { return stream; }
  [[nodiscard]] int get_debug_level() const {
    if (rank != 0)
      return PARMGMC_DEBUG_LEVEL_NONE;
    return debugLevel;
  }

  [[nodiscard]] int get_rank() const { return rank; }

private:
  logger_stream() :  stream{std::cout} {
    if (const char *dbgLevel = std::getenv("PARMGMC_DEBUG_LEVEL"))
      debugLevel = std::atoi(dbgLevel);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }

  int rank;
  int debugLevel{PARMGMC_DEBUG_LEVEL};
  std::ostream &stream;
};
}; // namespace parmgmc

#define PARMGMC_DEBUG_STREAM(level)                                            \
  if (level > ::parmgmc::logger_stream::get().get_debug_level())               \
    ;                                                                          \
  else                                                                         \
    ::parmgmc::logger_stream::get().get_stream()                               \
        << "[ParMGMC (" << parmgmc::logger_stream::get().get_rank() << ")] "

#define PARMGMC_INFO PARMGMC_DEBUG_STREAM(PARMGMC_DEBUG_LEVEL_INFO)
#define PARMGMC_DEBUG PARMGMC_DEBUG_STREAM(PARMGMC_DEBUG_LEVEL_VERBOSE)

#define PARMGMC_DEBUG_STREAM_NP(level)                                         \
  if (level > ::parmgmc::logger_stream::get().get_debug_level())               \
    ;                                                                          \
  else                                                                         \
    ::parmgmc::logger_stream::get().get_stream()

#define PARMGMC_INFO_NP PARMGMC_DEBUG_STREAM_NP(PARMGMC_DEBUG_LEVEL_INFO)
#define PARMGMC_DEBUG_NP PARMGMC_DEBUG_STREAM_NP(PARMGMC_DEBUG_LEVEL_VERBOSE)
