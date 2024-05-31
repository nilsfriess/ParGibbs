cmake_minimum_required(VERSION 3.20)
project(ParMGMC LANGUAGES C CXX)

set(CMAKE_COLOR_DIAGNOSTICS ON)

# find_package(MPI REQUIRED)
# set(MPI_TARGET MPI::MPI_C)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC PETSc>=3.17 IMPORTED_TARGET REQUIRED)

if (${PETSC_FOUND})
  message(STATUS "Found PETSc version ${PETSC_VERSION}, include dirs: ${PETSC_INCLUDE_DIRS}")
endif()

add_library(parmgmc SHARED
		    src/pc_hogwild.c
		    src/pc_gibbs.c
		    src/ziggurat.c
		    src/mc_sor.c
		    src/parmgmc.c
		    )
target_include_directories(parmgmc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(parmgmc SYSTEM PUBLIC ${PETSC_INCLUDE_DIRS})
target_compile_options(parmgmc PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(parmgmc PUBLIC ${PETSC_CFLAGS})
target_link_libraries(parmgmc PUBLIC PkgConfig::PETSC)

add_subdirectory(examples_c)
add_subdirectory(examples_cpp)

# if(NOT DEFINED PARMGMC_ENABLE_TESTS)
#   set(PARMGMC_ENABLE_TESTS TRUE)
# endif()

# if(${PARMGMC_ENABLE_TESTS})
#   add_subdirectory(tests)
# endif()

if(NOT DEFINED PARMGMC_ENABLE_PYTHON_BINDINGS)
  set(PARMGMC_ENABLE_PYTHON_BINDINGS False)
endif()

if(${PARMGMC_ENABLE_PYTHON_BINDINGS})
  add_subdirectory(python)
endif()
