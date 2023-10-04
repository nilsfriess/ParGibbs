cmake_minimum_required(VERSION 3.12)
project(ParGibbs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if (NOT DEFINED USE_MPI)
  add_subdirectory(external/FakeMPI)
  set(MPI_TARGET MPI_C)
else()
  find_package(MPI REQUIRED)
  message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
  set(MPI_TARGET MPI::MPI_CXX)
endif()

add_library(pargibbs INTERFACE)
target_include_directories(pargibbs INTERFACE include)
target_compile_options(pargibbs INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(pargibbs INTERFACE Eigen3::Eigen ${MPI_TARGET})

if (DEFINED USE_MPI)
  target_compile_definitions(pargibbs INTERFACE USE_MPI=${USE_MPI})

  if (NOT ${USE_MPI})
    target_include_directories(pargibbs INTERFACE SYSTEM external)
  endif()
endif()

add_subdirectory(src)
