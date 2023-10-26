cmake_minimum_required(VERSION 3.14)
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
  set(USE_MPI TRUE)
endif()

IF (${USE_MPI})
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

add_subdirectory(src)
