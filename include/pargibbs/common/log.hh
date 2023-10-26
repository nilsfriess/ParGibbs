#pragma once

#include <iostream>

#define PARGIBBS_DEBUG_LEVEL_NONE 0
#define PARGIBBS_DEBUG_LEVEL_INFO 1
#define PARGIBBS_DEBUG_LEVEL_VERBOSE 2

#ifndef PARGIBBS_DEBUG_LEVEL
#define PARGIBBS_DEBUG_LEVEL PARGIBBS_DEBUG_LEVEL_INFO
#endif

namespace pargibbs {
class logger_stream {
public:
  static logger_stream &get() {
    static logger_stream log;
    return log;
  }

  std::ostream &get_stream() const { return stream; }
  int get_debug_level() const { return debug_level; }

private:
  logger_stream() : debug_level{PARGIBBS_DEBUG_LEVEL}, stream{std::cout} {
    if (const char *dbg_level = std::getenv("PARGIBBS_DEBUG_LEVEL"))
      debug_level = std::atoi(dbg_level);
  }

  int debug_level;
  std::ostream &stream;
};
}; // namespace pargibbs

#define PARGIBBS_DEBUG_STREAM(level)                                           \
  if (level > ::pargibbs::logger_stream::get().get_debug_level())              \
    ;                                                                          \
  else                                                                         \
    ::pargibbs::logger_stream::get().get_stream() << "[ParGIBBS] "

#define PARGIBBS_DEBUG PARGIBBS_DEBUG_STREAM(PARGIBBS_DEBUG_LEVEL_INFO)

#define PARGIBBS_DEBUG_STREAM_NP(level)                                        \
  if (level > ::pargibbs::logger_stream::get().get_debug_level())              \
    ;                                                                          \
  else                                                                         \
    ::pargibbs::logger_stream::get().get_stream()

#define PARGIBBS_DEBUG_NP PARGIBBS_DEBUG_STREAM_NP(PARGIBBS_DEBUG_LEVEL_INFO)
