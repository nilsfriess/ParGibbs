#pragma once

#include <cstdlib>
#include <iostream>

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

  std::ostream &get_stream() const { return stream; }
  int get_debug_level() const { return debug_level; }

private:
  logger_stream() : debug_level{PARMGMC_DEBUG_LEVEL}, stream{std::cout} {
    if (const char *dbg_level = std::getenv("PARMGMC_DEBUG_LEVEL"))
      debug_level = std::atoi(dbg_level);
  }

  int debug_level;
  std::ostream &stream;
};
}; // namespace parmgmc

#define PARMGMC_DEBUG_STREAM(level)                                            \
  if (level > ::parmgmc::logger_stream::get().get_debug_level())               \
    ;                                                                          \
  else                                                                         \
    ::parmgmc::logger_stream::get().get_stream() << "[Parmgmc] "

#define PARMGMC_DEBUG PARMGMC_DEBUG_STREAM(PARMGMC_DEBUG_LEVEL_INFO)

#define PARMGMC_DEBUG_STREAM_NP(level)                                         \
  if (level > ::parmgmc::logger_stream::get().get_debug_level())               \
    ;                                                                          \
  else                                                                         \
    ::parmgmc::logger_stream::get().get_stream()

#define PARMGMC_DEBUG_NP PARMGMC_DEBUG_STREAM_NP(PARMGMC_DEBUG_LEVEL_INFO)
