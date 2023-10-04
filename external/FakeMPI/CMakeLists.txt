cmake_minimum_required(VERSION 3.15)

project(
  FakeMPI
  VERSION 3.1.0
  LANGUAGES C)

# Add the library
add_library(MPI_C unimpi.c experimental_unimpi.c)
set_property(TARGET MPI_C PROPERTY C_STANDARD 99)
