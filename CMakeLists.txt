cmake_minimum_required(VERSION 3.24)
project(ParMGMC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(MPI REQUIRED)
set(MPI_TARGET MPI::MPI_CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc>=3.4)

add_library(parmgmc INTERFACE)
target_include_directories(parmgmc INTERFACE "${CMAKE_SOURCE_DIR}/include")
target_compile_options(parmgmc INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(parmgmc INTERFACE PkgConfig::PETSc ${MPI_TARGET})

add_subdirectory(examples)
