include(FetchContent)
FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)

cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

add_library(pargibbs
  pargibbs/lattice/lattice.cc
  pargibbs/samplers/gibbs.cc)

target_include_directories(pargibbs PUBLIC ../include)
target_include_directories(pargibbs SYSTEM PUBLIC ${pcgcpp_INCLDUE_DIR})
target_compile_options(pargibbs PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(pargibbs PUBLIC Eigen3::Eigen ${MPI_TARGET})

if (${USE_MPI})
  target_compile_definitions(pargibbs PUBLIC USE_MPI=true)
else()
  target_compile_definitions(pargibbs PUBLIC USE_MPI=false)
  target_include_directories(pargibbs SYSTEM PUBLIC external)
endif()

if (${USE_METIS})
   target_compile_definitions(pargibbs PUBLIC USE_METIS=true)

   target_include_directories(pargibbs SYSTEM PUBLIC "${METIS_ROOT}/include/")
   target_link_directories(pargibbs PUBLIC "${METIS_ROOT}/lib")
   target_link_libraries(pargibbs PUBLIC metis GKlib)
endif()
