cmake_minimum_required(VERSION 3.24)
project(ParMGMC LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(MPI REQUIRED)
set(MPI_TARGET MPI::MPI_CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
find_package(PETSc REQUIRED)
if (${PETSC_FOUND})
  message(STATUS "Found PETSc version ${PETSC_VERSION}")
endif()

add_library(parmgmc INTERFACE)
target_include_directories(parmgmc INTERFACE "${CMAKE_SOURCE_DIR}/include" ${PETSC_INCLUDES})
target_compile_options(parmgmc INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(parmgmc INTERFACE ${PETSC_LIBRARIES} ${MPI_TARGET})

add_subdirectory(examples)
