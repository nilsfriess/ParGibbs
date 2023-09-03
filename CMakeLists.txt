cmake_minimum_required(VERSION 3.12)
project(ParGibbs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

add_library(pargibbs INTERFACE)
target_include_directories(pargibbs INTERFACE include)
target_compile_options(pargibbs INTERFACE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(pargibbs INTERFACE Eigen3::Eigen MPI::MPI_CXX)

add_subdirectory(tests)
add_subdirectory(src)
