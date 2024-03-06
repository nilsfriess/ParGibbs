cmake_minimum_required(VERSION 3.24)
project(ParMGMC LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(MPI REQUIRED)
set(MPI_TARGET MPI::MPI_CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC PETSc>=3.17 IMPORTED_TARGET REQUIRED)

if (${PETSC_FOUND})
  message(STATUS "Found PETSc version ${PETSC_VERSION}, include dirs: ${PETSC_INCLUDE_DIRS}")
endif()

add_library(parmgmc INTERFACE)
target_include_directories(parmgmc INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include" ${PETSC_INCLUDE_DIRS})
target_compile_options(parmgmc INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(parmgmc INTERFACE PkgConfig::PETSC ${MPI_TARGET})

add_subdirectory(examples)

if(NOT DEFINED PARMGMC_ENABLE_TESTS)
  set(PARMGMC_ENABLE_TESTS TRUE)
endif()

if(${PARMGMC_ENABLE_TESTS})
  add_subdirectory(tests)
endif()
