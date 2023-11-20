cmake_minimum_required(VERSION 3.24)
project(ParMGMC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_COLOR_DIAGNOSTICS ON)

if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DPG_DEBUG_MODE")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if (NOT DEFINED USE_MPI)
  set(USE_MPI TRUE)
endif()

if (${USE_MPI})
  find_package(MPI REQUIRED)
  set(MPI_TARGET MPI::MPI_CXX)
else()
  add_subdirectory(external/FakeMPI)
  set(MPI_TARGET MPI_C)
endif()

add_subdirectory(src)
add_subdirectory(examples)

if (NOT DEFINED ENABLE_TESTS)
  set(ENABLE_TESTS TRUE)
endif()

if (${ENABLE_TESTS})
  add_subdirectory(tests)
endif()
