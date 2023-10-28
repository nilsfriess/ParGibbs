cmake_minimum_required(VERSION 3.24)
project(ParGibbs)

set(CMAKE_CXX_STANDARD 20)
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

add_library(pargibbs INTERFACE)
target_include_directories(pargibbs INTERFACE include)
target_compile_options(pargibbs INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(pargibbs INTERFACE Eigen3::Eigen ${MPI_TARGET})

if (${USE_MPI})
  target_compile_definitions(pargibbs INTERFACE USE_MPI=true)
else()
  target_compile_definitions(pargibbs INTERFACE USE_MPI=false)
  target_include_directories(pargibbs INTERFACE SYSTEM external)
endif()

if (${USE_METIS})
   target_compile_definitions(pargibbs INTERFACE USE_METIS=true)

   target_include_directories(pargibbs INTERFACE SYSTEM "${METIS_ROOT}/include/")
   target_link_directories(pargibbs INTERFACE "${METIS_ROOT}/lib")
   target_link_libraries(pargibbs INTERFACE metis GKlib)
endif()

add_subdirectory(src)
